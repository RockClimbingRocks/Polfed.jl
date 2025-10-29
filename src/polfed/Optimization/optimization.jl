
const Offdiagonal   = Tuple{<:Real, Vector{Int}, Vector{Int}}
const Offdiagonals = Union{Offdiagonal, Vector{<:Offdiagonal}}


include("cpu/cpu.jl")



function optimize_spectral_transform(mat::AbstractMatrix{T}, spectral_transform::SpectralTransformConfig) where {T<:Real}
    # Implement optimization logic here
    parallel_strategy = spectral_transform.parallelization
    diagonals, offdiagonals = get_diags_and_offdiagonals_by_value(mat)

    pu = isa(mat, CuArray) ? GPU() : CPU()
    v0 = pu.rand(size(mat,1)); v0 ./= norm(v0)
    f!_opt = optimized_mapping!(diagonals, offdiagonals, parallel_strategy)

    Emin = first(collect(lanczos(f!_opt, v0, 1; which=:SR, maxdim=1000)[1]))
    Emax = last(collect(lanczos(f!_opt, v0, 1; which=:LR,  maxdim=1000)[1]))
    spread = (Emax-Emin)/(2-1e-12)
    center = (Emax+Emin)/2.
    # spread = (Emax-Emin)/2.

    diagonals_rescaled = (diagonals .- center)./spread
    offdiagonals_rescaled = [(val/spread, flat, starts) for (val, flat, starts) in offdiagonals]
    offdiagonals_rescaled = length(offdiagonals_rescaled)==1 ? offdiagonals_rescaled[1] : offdiagonals_rescaled



    f!_opt_rescaled = optimized_mapping!(diagonals_rescaled, offdiagonals_rescaled, parallel_strategy)
    crr_opt_rescaled = optimized_clenshaw_recurrence_relation!(diagonals_rescaled, offdiagonals_rescaled, parallel_strategy)
    cfs_opt_rescaled = optimized_clenshaw_final_sum!(diagonals_rescaled, offdiagonals_rescaled, parallel_strategy)


    spectral_transform.f!_rescaled = f!_opt_rescaled
    spectral_transform.clenshaw_recurrence = crr_opt_rescaled
    spectral_transform.clenshaw_finalsum = cfs_opt_rescaled
    spectral_transform.Emin = Emin
    spectral_transform.Emax = Emax

    return f!_opt
end



function get_diags_and_offdiagonals_by_value(mat::AbstractMatrix{T}; tol=1e-13, round_digits=15) where {T<:Real}
    dim = size(mat, 1)

    # Map: val => list of connections for each row
    value_to_conn = Dict{Float64, Vector{Vector{Int}}}()

    # Collect diagonal entries
    diagonals = Vector{Float64}(undef, dim)

    for i in 1:dim
        diagonals[i] = round(mat[i, i]; digits=round_digits)  # store diagonal value

        row_conns = Dict{Float64, Vector{Int}}()
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            if i == j  # skip diagonal
                continue
            end
            v = mat[i, j]
            if abs(v) < tol
                continue
            end
            v_rounded = round(v; digits=round_digits)
            push!(get!(row_conns, v_rounded, Int[]), j)
        end
        for (v, js) in row_conns
            if !haskey(value_to_conn, v)
                value_to_conn[v] = [Int[] for _ in 1:dim]
            end
            value_to_conn[v][i] = js
        end
    end

    # Now flatten each list
    offdiagonals = Tuple{Float64, Vector{Int}, Vector{Int}}[]
    for (v, conn_lists) in sort(collect(value_to_conn), by=first)
        flat = Int[]
        starts = Int[]
        idx = 1
        for js in conn_lists
            push!(starts, idx)
            append!(flat, js)
            idx += length(js)
        end
        push!(offdiagonals, (v, flat, starts))
    end

    # length(offdiagonals)==1 && (return diagonals, offdiagonals[1])
    return diagonals, offdiagonals
end



