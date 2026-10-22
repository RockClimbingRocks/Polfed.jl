# Off-diagonal structure: what a Hamiltonian's off-diagonals ARE, independent of which kernel
# will consume them.
#
# The row kernels never read a generic sparse matrix. They read a compact, value-grouped
# description: the diagonal, plus one group per DISTINCT off-diagonal value, each group holding
# its neighbour columns in CSR form. That representation is deliberately general:
#
#   • several distinct off-diagonal values  → several groups
#   • rows of differing length (symmetry sectors!) → per-row extents inside each group
#
# so `classify` never refuses a structured Hamiltonian. It reports the facts; each pattern folder
# under patterns/ then decides for itself whether its kernels can run on them.

"""
    ValueGroup{T,TI}

One distinct off-diagonal value together with the neighbour columns carrying it, in CSR form.

# Fields
- `value::T`: the shared off-diagonal value.
- `flat::Vector{TI}`: neighbour columns, concatenated row by row.
- `starts::Vector{Int}`: length `dim + 1`. Row `i`'s neighbours in this group are
  `flat[starts[i] : starts[i+1] - 1]`, so a row with none is simply an empty range and the last
  row needs no special case.

Row extents live in `starts`, which is what lets a group hold **irregular** rows.
"""
struct ValueGroup{T<:Number, TI<:Integer}
    value::T
    flat::Vector{TI}
    starts::Vector{Int}
end

"""    nnz_per_row(g::ValueGroup, i::Int) -> Int — how many neighbours row `i` has in this group."""
@inline nnz_per_row(g::ValueGroup, i::Int) = g.starts[i + 1] - g.starts[i]

"""
    MappingStructure{T,TI}

The value-grouped description of a Hamiltonian that the row kernels consume, plus the facts
[`mapping_kernels`](@ref) needs to decide which kernels apply.

# Fields
- `dim::Int`: Hilbert-space dimension `D`.
- `diags::Vector{T}`: the diagonal, length `D`.
- `groups::Vector{ValueGroup{T,TI}}`: one entry per distinct off-diagonal value.
- `lmin::Int`, `lmax::Int`: fewest / most off-diagonal neighbours over all rows (summed across
  groups). `lmin == lmax` means every row has the same number of neighbours.
"""
struct MappingStructure{T<:Number, TI<:Integer}
    dim::Int
    diags::Vector{T}
    groups::Vector{ValueGroup{T,TI}}
    lmin::Int
    lmax::Int
end

Base.eltype(::Type{MappingStructure{T,TI}}) where {T,TI} = T

"""    n_values(s::MappingStructure) -> Int — number of distinct off-diagonal values."""
n_values(s::MappingStructure) = length(s.groups)

"""    is_regular(s::MappingStructure) -> Bool — do all rows carry the same neighbour count?"""
is_regular(s::MappingStructure) = s.lmin == s.lmax

"""
    _round_key(v, round_digits) -> value used to group off-diagonals

Off-diagonals that agree to `round_digits` decimals are treated as one value. Rounding rather
than exact comparison keeps a Hamiltonian assembled in floating point from splitting one physical
coupling into a dozen numerically distinct "values".
"""
_round_key(v::Number, round_digits::Int) = round(v; digits = round_digits)

"""
    classify(H::AbstractMatrix; tol=1e-12, round_digits=12) -> MappingStructure

Extract the value-grouped structure of `H`: the diagonal, and one [`ValueGroup`](@ref) per
distinct off-diagonal value.

Unlike a strict extractor this **does not refuse** irregular rows or several off-diagonal values.
Both are representable — several values become several groups, and differing row lengths live in
each group's `starts` — so any structured Hamiltonian can be described here and run by the
universal kernel.

# Arguments
- `tol`: entries with `|value| < tol` are structural zeros and are dropped.
- `round_digits`: off-diagonals agreeing to this many decimals count as one value.

# Throws
- `ArgumentError` if `H` is not square, has no off-diagonal entries, or exceeds the `Int32`
  index range.
"""
function classify(H::AbstractMatrix{T}; tol::Real = 1e-12, round_digits::Int = 12) where {T<:Number}
    dim = size(H, 1)
    dim == size(H, 2) ||
        throw(ArgumentError("classify: H must be square, got $(size(H))."))
    dim <= typemax(Int32) ||
        throw(ArgumentError("classify: dim = $dim exceeds the Int32 index range."))

    diags = zeros(T, dim)

    # Counting sort into CSR, in two passes over the nonzeros: pass 1 counts entries per
    # (group, row), pass 2 places them. Allocation is 4 arrays PER GROUP — not per row.
    #
    # The previous version kept a `Vector{Vector{Int32}}` of per-row lists per value and did
    #     rows = get!(cols_by_value, key, _emptyrows())
    # `get!` evaluates its default EAGERLY, so `_emptyrows()` built a fresh dim-length vector
    # of empty vectors for EVERY nonzero, not just for each new value. On a 15-value operator
    # at L=16 that was 29.6 GB of allocation and minutes of GC for a matrix of a few hundred MB.
    group_of = Dict{T,Int}()          # rounded value → group index
    values_seen = T[]                 # group index → its value
    counts = Vector{Int}[]            # group index → per-row entry count, later the cursor

    for (i, j, v) in _entries(H)
        abs(v) < tol && continue
        if i == j
            diags[i] = v
            continue
        end
        key = _round_key(v, round_digits)
        g = get(group_of, key, 0)
        if g == 0                     # a value seen for the first time: one allocation
            push!(values_seen, key)
            push!(counts, zeros(Int, dim))
            g = length(values_seen)
            group_of[key] = g
        end
        @inbounds counts[g][i] += 1
    end

    isempty(values_seen) &&
        throw(ArgumentError("classify: H has no off-diagonal entries above tol = $tol."))

    ng = length(values_seen)
    starts_of = Vector{Vector{Int}}(undef, ng)
    flat_of = Vector{Vector{Int32}}(undef, ng)
    for g in 1:ng
        c = counts[g]
        st = Vector{Int}(undef, dim + 1)
        st[1] = 1
        @inbounds for i in 1:dim
            st[i + 1] = st[i] + c[i]
        end
        starts_of[g] = st
        flat_of[g] = Vector{Int32}(undef, st[end] - 1)
        copyto!(c, 1, st, 1, dim)     # the count array becomes the write cursor
    end

    # Pass 2 walks `_entries` in the same order as pass 1, so each row's neighbours land in the
    # order the matrix yields them (ascending column for a `SparseMatrixCSC`) — the same order
    # the old implementation produced.
    for (i, j, v) in _entries(H)
        abs(v) < tol && continue
        i == j && continue
        g = group_of[_round_key(v, round_digits)]
        @inbounds begin
            c = counts[g]
            flat_of[g][c[i]] = Int32(j)
            c[i] += 1
        end
    end

    # stable order: ascending value, so the same H always produces the same group order
    groups = [ValueGroup{T,Int32}(values_seen[g], flat_of[g], starts_of[g])
              for g in sortperm(values_seen; by = real)]

    # total neighbours per row, summed over groups
    lmin, lmax = typemax(Int), 0
    for i in 1:dim
        n = 0
        for g in groups
            n += nnz_per_row(g, i)
        end
        lmin = min(lmin, n)
        lmax = max(lmax, n)
    end

    return MappingStructure{T,Int32}(dim, diags, groups, lmin, lmax)
end

"""    _entries(H) -> iterator of (row, col, value) — sparse fast path, dense fallback."""
function _entries(H::SparseMatrixCSC{T}) where {T}
    rows = rowvals(H)
    vals = nonzeros(H)
    return ((rows[k], j, vals[k]) for j in 1:size(H, 2) for k in nzrange(H, j))
end
_entries(H::AbstractMatrix) =
    ((i, j, H[i, j]) for j in 1:size(H, 2) for i in 1:size(H, 1))

"""
    rescale(s::MappingStructure, a::Real, b::Real) -> MappingStructure

The structure of the rescaled operator ``\\tilde H = (H - b)/a``: shift and scale the diagonal,
scale every group value. The neighbour tables are shared with `s` — rescaling never moves an
off-diagonal, so the index arrays are reused rather than copied.
"""
function rescale(s::MappingStructure{T,TI}, a::Real, b::Real) where {T,TI}
    inv_a = one(T) / T(a)
    diags = @. (s.diags - T(b)) * inv_a
    groups = [ValueGroup{T,TI}(g.value * inv_a, g.flat, g.starts) for g in s.groups]
    return MappingStructure{T,TI}(s.dim, diags, groups, s.lmin, s.lmax)
end

"""
    show(io, ::MIME"text/plain", s::MappingStructure)

Human summary: dimension, value count, and the row-length range — the facts that decide which
kernels apply.
"""
function Base.show(io::IO, ::MIME"text/plain", s::MappingStructure)
    println(io, "MappingStructure")
    println(io, "  dim        : ", s.dim)
    println(io, "  values     : ", n_values(s), " (", join((string(g.value) for g in s.groups), ", "), ")")
    print(io,   "  row length : ", s.lmin == s.lmax ? string(s.lmin, " (regular)") :
                                    string(s.lmin, " … ", s.lmax, " (irregular)"))
    return nothing
end
