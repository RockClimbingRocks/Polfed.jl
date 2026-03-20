
const Offdiagonal   = Tuple{<:Number, Vector{Int}, Vector{Int}}
const Offdiagonals = Union{Offdiagonal, Vector{<:Offdiagonal}}


include("cpu/cpu.jl")

"""
    round_key(x, digits) -> Number

Round real/complex values to stable grouping keys used when bucketing matrix
entries by value.
"""
@inline round_key(x::Real, digits::Int) = round(x; digits=digits)
@inline round_key(x::Complex, digits::Int) = complex(round(real(x); digits=digits), round(imag(x); digits=digits))

"""
    sort_key(x) -> Tuple

Sort helper for deterministic ordering of real/complex value buckets.
"""
@inline sort_key(x::Real) = (x, zero(x))
@inline sort_key(x::Complex) = (real(x), imag(x))

"""
    _extrema_probe_tolerances(::Type{T}) where {T<:Number} -> Tuple{Real,Real}

Return `(tol, eigentol)` used for robust extremal Ritz-value probing.
"""
@inline function _extrema_probe_tolerances(::Type{T}) where {T<:Number}
    tol = sqrt(eps(real(T)))
    return tol, tol
end

"""
    _get_extremal_ritz_value(f!::Function, v0::AbstractVector, which::Symbol) -> Number

Estimate an extremal eigenvalue (`:SR` or `:LR`) via a short Lanczos probe.
"""
function _get_extremal_ritz_value(f!::Function, v0::AbstractVector, which::Symbol)
    tol, eigentol = _extrema_probe_tolerances(eltype(v0))
    vals = collect(lanczos(f!, v0, 1; which=which, maxdim=1000, tol=tol, eigentol=eigentol)[1])
    isempty(vals) && error(
        "Failed to estimate $(which) extremal eigenvalue during mapping optimization. " *
        "Try setting `MappingConfig(Emin=..., Emax=...)` explicitly or relax factorization tolerances."
    )
    return which === :SR ? first(vals) : last(vals)
end


"""
    optimize_spectral_transform(mat::AbstractMatrix{T}, mapping::MappingConfig) where {T<:Number} -> Function

Precompute optimized mapping and Clenshaw kernels for value-grouped matrices.

This function:
1. groups diagonal/off-diagonal entries by value,
2. builds fast mapping callbacks,
3. estimates `Emin/Emax`,
4. builds rescaled optimized callbacks and stores them into `mapping`.

# Returns
- Unscaled optimized mapping callback `f!_opt`.
"""
function optimize_spectral_transform(mat::AbstractMatrix{T}, mapping::MappingConfig) where {T<:Number}
    # Implement optimization logic here
    if is_gpu_array(mat) && !(eltype(mat) <: Real)
        error("Complex GPU optimization is not supported yet. Current GPU optimization kernels are real-only. Use CPU arrays.")
    end
    parallel_strategy = mapping.parallel_strategy
    if parallel_strategy === nothing
        parallel_strategy = is_gpu_array(mat) ? NoParallel() : MulColsParallel()
        mapping.parallel_strategy = parallel_strategy
    end
    diagonals, offdiagonals = get_diags_and_offdiagonals_by_value(mat)

    pu = is_gpu_array(mat) ? GPU() : CPU()
    v0 = pu.randn(eltype(mat), size(mat, 1))
    v0 ./= norm(v0)
    f!_opt = optimized_mapping!(diagonals, offdiagonals, parallel_strategy)

    Emin = _get_extremal_ritz_value(f!_opt, v0, :SR)
    Emax = _get_extremal_ritz_value(f!_opt, v0, :LR)
    spread = (Emax-Emin)/2
    center = (Emax+Emin)/2

    diagonals_rescaled = (diagonals .- center)./spread
    offdiagonals_rescaled = [(val/spread, flat, starts) for (val, flat, starts) in offdiagonals]
    offdiagonals_rescaled = length(offdiagonals_rescaled)==1 ? offdiagonals_rescaled[1] : offdiagonals_rescaled



    f!_opt_rescaled = optimized_mapping!(diagonals_rescaled, offdiagonals_rescaled, parallel_strategy)
    crr_opt_rescaled = optimized_clenshaw_recurrence_relation!(diagonals_rescaled, offdiagonals_rescaled, parallel_strategy)
    cfs_opt_rescaled = optimized_clenshaw_final_sum!(diagonals_rescaled, offdiagonals_rescaled, parallel_strategy)


    mapping.f!_rescaled = f!_opt_rescaled
    mapping.clenshaw_recurrence = crr_opt_rescaled
    mapping.clenshaw_finalsum = cfs_opt_rescaled
    mapping.Emin = Emin
    mapping.Emax = Emax

    return f!_opt
end



"""
    get_diags_and_offdiagonals_by_value(mat; tol=1e-13, round_digits=15)

Extract diagonal values and group off-diagonal column indices by rounded value.

Supported for both dense `AbstractMatrix` and `SparseMatrixCSC`.

# Returns
- `(diagonals, offdiagonals)` where `offdiagonals` is a packed representation
  suitable for [`optimized_mapping!`](@ref).
"""
function get_diags_and_offdiagonals_by_value(mat::SparseMatrixCSC{T}; tol=1e-13, round_digits=15) where {T<:Number}
    dim = size(mat, 1)

    K = typeof(round_key(zero(T), round_digits))
    diagonals = Vector{K}(undef, dim)
    value_to_counts = Dict{K, Vector{Int}}()

    rows = rowvals(mat)
    nz = nonzeros(mat)

    # First pass: collect diagonals and per-row counts for each unique off-diagonal value.
    @inbounds for i in 1:dim
        diagonals[i] = round_key(mat[i, i], round_digits)

        for col in nzrange(mat, i)
            j = rows[col]
            i == j && continue
            v = nz[col]
            abs(v) < tol && continue
            key = round_key(v, round_digits)
            counts = get!(value_to_counts, key) do
                zeros(Int, dim)
            end
            counts[i] += 1
        end
    end

    value_to_flat, value_to_starts = allocate_offdiagonal_buffers(value_to_counts, dim)

    # Second pass: fill flattened indices in preallocated buffers.
    @inbounds for i in 1:dim
        for col in nzrange(mat, i)
            j = rows[col]
            i == j && continue
            v = nz[col]
            abs(v) < tol && continue
            key = round_key(v, round_digits)
            next_indices = value_to_counts[key]
            flat = value_to_flat[key]
            idx = next_indices[i]
            flat[idx] = j
            next_indices[i] = idx + 1
        end
    end

    return diagonals, pack_offdiagonals(value_to_flat, value_to_starts)
end

"""Dense-matrix overload of `get_diags_and_offdiagonals_by_value`."""
function get_diags_and_offdiagonals_by_value(mat::AbstractMatrix{T}; tol=1e-13, round_digits=15) where {T<:Number}
    dim = size(mat, 1)

    K = typeof(round_key(zero(T), round_digits))
    diagonals = Vector{K}(undef, dim)
    value_to_counts = Dict{K, Vector{Int}}()

    # First pass: collect diagonals and per-row counts for each unique off-diagonal value.
    @inbounds for i in 1:dim
        diagonals[i] = round_key(mat[i, i], round_digits)

        for j in 1:dim
            i == j && continue
            v = mat[i, j]
            abs(v) < tol && continue
            key = round_key(v, round_digits)
            counts = get!(value_to_counts, key) do
                zeros(Int, dim)
            end
            counts[i] += 1
        end
    end

    value_to_flat, value_to_starts = allocate_offdiagonal_buffers(value_to_counts, dim)

    # Second pass: fill flattened indices in preallocated buffers.
    @inbounds for i in 1:dim
        for j in 1:dim
            i == j && continue
            v = mat[i, j]
            abs(v) < tol && continue
            key = round_key(v, round_digits)
            next_indices = value_to_counts[key]
            flat = value_to_flat[key]
            idx = next_indices[i]
            flat[idx] = j
            next_indices[i] = idx + 1
        end
    end

    return diagonals, pack_offdiagonals(value_to_flat, value_to_starts)
end

"""
    allocate_offdiagonal_buffers(value_to_counts::Dict{K, Vector{Int}}, dim::Int) where {K}

Allocate flattened off-diagonal storage and row-start offsets for each unique
off-diagonal value bucket.
"""
function allocate_offdiagonal_buffers(value_to_counts::Dict{K, Vector{Int}}, dim::Int) where {K}
    value_to_flat = Dict{K, Vector{Int}}()
    value_to_starts = Dict{K, Vector{Int}}()

    for (v, counts) in value_to_counts
        starts = Vector{Int}(undef, dim)
        total = 0
        @inbounds for i in 1:dim
            starts[i] = total + 1
            total += counts[i]
        end
        value_to_flat[v] = Vector{Int}(undef, total)
        value_to_starts[v] = starts
        counts .= starts
    end

    return value_to_flat, value_to_starts
end

"""
    pack_offdiagonals(value_to_flat::Dict{K, Vector{Int}}, value_to_starts::Dict{K, Vector{Int}}) where {K}

Convert intermediate dictionaries into sorted packed off-diagonal tuples
`(value, flat_indices, row_starts)`.
"""
function pack_offdiagonals(value_to_flat::Dict{K, Vector{Int}}, value_to_starts::Dict{K, Vector{Int}}) where {K}
    offdiagonals = Tuple{K, Vector{Int}, Vector{Int}}[]
    for v in sort(collect(keys(value_to_flat)); by=sort_key)
        push!(offdiagonals, (v, value_to_flat[v], value_to_starts[v]))
    end
    return offdiagonals
end
