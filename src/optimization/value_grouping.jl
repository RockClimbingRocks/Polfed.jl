# Value grouping: extract the fast-kernel structure from a (structured) Hamiltonian.
#
# The row kernels do not read a generic sparse matrix; they consume a compact, regular
# description of the off-diagonals. Stage 1 targets the regime these kernels were tuned for
# (and the regime where value-grouped optimization pays off): a SINGLE distinct off-diagonal
# value `hx` and a REGULAR row length `L` (every row has exactly `L` off-diagonal neighbours) —
# e.g. QREM / uniform-σˣ models. Multi-value / irregular Hamiltonians fall back to the generic
# Clenshaw path (`optimize_mapping = false`) until a multi-group kernel is available.

"""
    MappingStructure{T}

Compact off-diagonal description consumed by the row-layout kernels ([`build_row_mapping`](@ref)).
Assumes a single off-diagonal value and a regular row length (see [`value_grouping`](@ref)).

# Fields
- `dim::Int`: Hilbert-space dimension `D`.
- `L::Int`: number of off-diagonal neighbours per row (regular).
- `hx::T`: the single off-diagonal value.
- `diags::Vector{T}`: the diagonal, length `D`.
- `nb32::Vector{Int32}`: ELL neighbour columns (1-based), `nb32[(i-1)*L + c]` = the `c`-th
  neighbour of row `i`. Int32 halves the index traffic (valid for `L ≤ 30`, i.e. `D ≤ 2³⁰`).
- `flat::Vector{Int}`, `starts::Vector{Int}`: CSR form of the same neighbours — `flat` holds the
  columns, `starts[i]` is row `i`'s (1-based) offset into `flat`. For the regular structure here
  `starts[i] = (i-1)*L + 1`, but the CSR kernels stay general.
"""
struct MappingStructure{T<:Number}
    dim::Int
    L::Int
    hx::T
    diags::Vector{T}
    nb32::Vector{Int32}
    flat::Vector{Int}
    starts::Vector{Int}
end

Base.eltype(::Type{MappingStructure{T}}) where {T} = T

"""
    value_grouping(H::AbstractMatrix; tol=1e-12, rtol=1e-9) -> MappingStructure

Extract the [`MappingStructure`](@ref) the row kernels need from `H`: the diagonal, the single
off-diagonal value, and the neighbour columns in both ELL (Int32) and CSR forms.

# Arguments
- `H`: a square Hamiltonian, `SparseMatrixCSC` (fast path) or a dense `AbstractMatrix`.
- `tol`: entries with `|value| < tol` are treated as structural zeros (dropped).
- `rtol`: relative tolerance for the "single off-diagonal value" check.

# Throws
- `ArgumentError`: if `H` is not square, has more than one distinct off-diagonal value, or has an
  irregular row length. These are the Stage-1 assumptions — use `optimize_mapping = false` for the
  generic path otherwise.
"""
function value_grouping(H::AbstractMatrix{T}; tol::Real = 1e-12, rtol::Real = 1e-9) where {T<:Number}
    dim = size(H, 1)
    dim == size(H, 2) || throw(ArgumentError("value_grouping: H must be square, got $(size(H))."))

    diags = zeros(T, dim)
    rowcols = [Int[] for _ in 1:dim]
    hx = zero(T)
    have_hx = false

    _collect_offdiags!(diags, rowcols, H, tol) do v
        # capture the first off-diagonal value; check the rest against it
        if !have_hx
            hx = v
            have_hx = true
        elseif !isapprox(v, hx; rtol = rtol, atol = tol)
            throw(ArgumentError(
                "value_grouping (Stage 1) supports a single off-diagonal value; found a second " *
                "distinct value ($v vs $hx). Use optimize_mapping = false for the generic path."))
        end
    end

    have_hx || throw(ArgumentError("value_grouping: H has no off-diagonal entries above tol=$tol."))

    L = length(rowcols[1])
    all(length(rc) == L for rc in rowcols) || throw(ArgumentError(
        "value_grouping (Stage 1) requires a regular row length (every row the same number of " *
        "off-diagonal neighbours). Use optimize_mapping = false for the generic path."))
    L ≤ 30 || throw(ArgumentError("value_grouping: L=$L exceeds the Int32-ELL limit (L ≤ 30)."))

    nb32   = Vector{Int32}(undef, dim * L)
    flat   = Vector{Int}(undef, dim * L)
    starts = Vector{Int}(undef, dim)
    @inbounds for i in 1:dim
        starts[i] = (i - 1) * L + 1
        rc = rowcols[i]
        for c in 1:L
            j = rc[c]
            nb32[(i - 1) * L + c] = Int32(j)
            flat[(i - 1) * L + c] = j
        end
    end

    return MappingStructure{T}(dim, L, hx, diags, nb32, flat, starts)
end

"""
    _collect_offdiags!(on_value, diags, rowcols, H, tol)

Fill `diags[i] = H[i,i]` and `push!(rowcols[i], j)` for every off-diagonal `H[i,j]` with
`|H[i,j]| ≥ tol`, calling `on_value(H[i,j])` on each so the caller can validate the single-value
assumption. `SparseMatrixCSC` fast path + a dense fallback.
"""
function _collect_offdiags!(on_value, diags, rowcols, H::SparseMatrixCSC{T}, tol) where {T}
    rows = rowvals(H)
    nz = nonzeros(H)
    @inbounds for j in 1:size(H, 2)
        for k in nzrange(H, j)
            i = rows[k]
            v = nz[k]
            if i == j
                diags[i] = v
            elseif abs(v) ≥ tol
                push!(rowcols[i], j)
                on_value(v)
            end
        end
    end
    return nothing
end

function _collect_offdiags!(on_value, diags, rowcols, H::AbstractMatrix{T}, tol) where {T}
    @inbounds for i in 1:size(H, 1)
        diags[i] = H[i, i]
        for j in 1:size(H, 2)
            i == j && continue
            v = H[i, j]
            if abs(v) ≥ tol
                push!(rowcols[i], j)
                on_value(v)
            end
        end
    end
    return nothing
end
