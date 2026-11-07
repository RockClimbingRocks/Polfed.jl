# HopStructure — what a periodic hopping operator IS, in the only form its kernels need.
#
#   H = diag(ε) + t · Σ_{axes k} (T_k^+ + T_k^-)        on a periodic D-dimensional lattice
#   y[i] = ε[i]·x[i] + t · Σ_{k=1}^{D} ( x[i + ê_k] + x[i - ê_k] )        (indices mod L_k)
#
# This is the Anderson model: on-site disorder plus uniform nearest-neighbour hopping. The
# neighbour is ARITHMETIC — site index ± a stride, with one wrap test per axis — so like
# `spin_flip` this pattern stores NO index table. `MappingStructure` (value-grouped CSR) would
# hold D·2·4 bytes per site of information already implied by the shape tuple: 168 MB at 128³,
# and the matrix it is extracted from is larger still.
#
# There are deliberately TWO ways to get one, and the second is the bigger win:
#
#   1. from a `MappingStructure` — `_derive_hop`, so the existing matrix entrypoints pick the
#      pattern up with no API change;
#   2. directly from `(diags, shape, hopping)` — no `H` is ever assembled. A 128³ Anderson
#      matrix is ~1.1 GB of `SparseMatrixCSC` purely to be thrown away after `classify`.
#
# Site linearisation is x-FASTEST (column-major over the lattice): site(x₁,…,x_D) =
# Σ_k x_k·s_k with s_1 = 1, s_{k+1} = s_k·L_k. Both kernels slide along axis 1, so axis 1 is
# the one whose neighbours are free; put the longest axis there if the lattice is anisotropic.

"""
    HopStructure{T,D}

A periodic nearest-neighbour hopping operator on a `D`-dimensional rectangular lattice.

# Fields
- `dim::Int`: number of sites, `prod(shape)`.
- `shape::NTuple{D,Int}`: lattice extents `(L₁, …, L_D)`, axis 1 fastest-varying.
- `strides::NTuple{D,Int}`: site-index strides, `s₁ = 1`, `s_{k+1} = s_k · L_k`. Derived, kept
  because every kernel needs them per line and recomputing a `prod` there is pure waste.
- `diags::Vector{T}`: the on-site energies, length `dim`.
- `hopping::T`: the single amplitude carried by every one of the `2D` bonds.

`D` is a TYPE parameter, not a field. That is the load-bearing decision of this pattern: a
measured runtime-dimensional kernel (extents in a `Vector`, a loop over axes in the hot path)
was 3.7-5.7x slower than the specialised one, so the dimension has to be known to the compiler
before the apply loop starts.

See [`HopStructure(diags, shape, hopping)`](@ref) for the validation rules.
"""
struct HopStructure{T<:Number, D}
    dim::Int
    shape::NTuple{D,Int}
    strides::NTuple{D,Int}
    diags::Vector{T}
    hopping::T
end

"""
    HopStructure(diags, shape, hopping) -> HopStructure

Build a periodic hopping operator directly from its lattice — **no matrix required**.

```julia
shape = (128, 128, 128)
hs = HopStructure(randn(prod(shape)), shape, -1.0)     # 3-D Anderson, W = 0 disorder scale
```

`shape` is any iterable of extents; the element type comes from `diags`, and `hopping` is
converted to it.

# Throws
- `ArgumentError` if `shape` is empty, if `prod(shape) != length(diags)`, or if any extent is
  below 3.

# Notes
Extents of 1 and 2 are rejected rather than supported. On a periodic axis of length 2 the `+ê`
and `-ê` neighbours are the SAME site, so the operator this pattern computes carries `2t` on
that bond while the matrix a user would write down carries `t` — the kernels would be right by
one convention and silently wrong by the other. Length 1 makes a site its own neighbour. Both
are degenerate lattices, not optimisation targets, so they are refused at construction where the
message is legible.
"""
function HopStructure(diags::Vector{T}, shape, hopping) where {T<:Number}
    dims = Int[Int(s) for s in shape]
    isempty(dims) && throw(ArgumentError(
        "HopStructure: the lattice needs at least one axis, got an empty shape."))
    all(>=(3), dims) || throw(ArgumentError(
        "HopStructure: every lattice extent must be at least 3, got $(Tuple(dims)). " *
        "On a periodic axis of length 2 the two neighbours coincide and the bond would be " *
        "double-counted; length 1 makes a site its own neighbour."))
    D = length(dims)
    dim = prod(dims)
    dim == length(diags) || throw(ArgumentError(
        "HopStructure: prod(shape) = $dim but length(diags) = $(length(diags))."))
    shp = ntuple(k -> dims[k], D)
    strd = ntuple(k -> k == 1 ? 1 : prod(@view dims[1:k-1]), D)
    return HopStructure{T,D}(dim, shp, strd, diags, convert(T, hopping))
end

Base.eltype(::Type{HopStructure{T,D}}) where {T,D} = T

"""    n_axes(hs::HopStructure) -> Int — the lattice dimension `D` (each axis contributes 2 bonds)."""
n_axes(::HopStructure{T,D}) where {T,D} = D

"""
    rescale(hs::HopStructure, a, b) -> HopStructure

The structure of ``\\tilde H = (H - b)/a``: shift and scale the on-site energies, scale the
hopping. The shape is untouched — rescaling never moves a bond.
"""
function rescale(hs::HopStructure{T,D}, a::Real, b::Real) where {T,D}
    inv_a = one(T) / T(a)
    return HopStructure{T,D}(hs.dim, hs.shape, hs.strides,
                             @.((hs.diags - T(b)) * inv_a), hs.hopping * inv_a)
end

"""
    _derive_hop(s::MappingStructure) -> HopStructure or nothing

Recover the lattice from a value-grouped structure, or `nothing` if `s` is not a periodic
hopping operator. This is what lets `mapping_kernels(H)` report the pattern and what lets the
existing matrix entrypoints use it with no API change.

Returns `nothing` (rather than throwing) because this is the question `can_apply` asks.

# How the shape is recovered
Row 1 is site 0, whose `2D` neighbour offsets are exactly

    s₁, s₁(L₁-1), s₂, s₂(L₂-1), …, s_D, s_D(L_D-1)

and — because `s_{k+1} = s_k·L_k` and every extent is at least 3 — that list is already in
ASCENDING order when read as written. So the sorted offsets of row 1 pair off left to right:
`(s_k, s_k(L_k-1))` gives `L_k = s_k(L_k-1) ÷ s_k + 1`, and `s_{k+1}` must be the next entry.
Every rounding of that reconstruction is then checked, and finally every row is verified to
generate exactly its own `2D` neighbours — one pass over the nonzeros, `O(nnz)`, the same order
as `classify` itself.

Extents below 3 break the ordering argument (`s_k(L_k-1) ≤ s_k`), which is the other reason
[`HopStructure`](@ref) refuses them.
"""
function _derive_hop(s::MappingStructure{T,TI}) where {T,TI}
    length(s.groups) == 1 || return nothing        # one uniform hopping amplitude, by definition
    g = s.groups[1]
    dim = s.dim
    s.lmin == s.lmax || return nothing             # every site has the same 2D bonds

    n = nnz_per_row(g, 1)
    (n >= 2 && iseven(n)) || return nothing
    nax = n ÷ 2
    offs = sort!(Int[Int(g.flat[c]) - 1 for c in g.starts[1]:(g.starts[2] - 1)])
    length(offs) == n || return nothing

    shape = Vector{Int}(undef, nax)
    strides = Vector{Int}(undef, nax)
    for k in 1:nax
        sk, tk = offs[2k - 1], offs[2k]
        sk == (k == 1 ? 1 : strides[k - 1] * shape[k - 1]) || return nothing
        (tk > sk && rem(tk, sk) == 0) || return nothing
        Lk = div(tk, sk) + 1
        Lk >= 3 || return nothing
        strides[k] = sk
        shape[k] = Lk
    end
    strides[nax] * shape[nax] == dim || return nothing

    # Every row must generate exactly its own 2·nax neighbours, once each.
    expected = Vector{Int}(undef, n)
    used = falses(n)
    for i in 1:dim
        p = i - 1
        (g.starts[i + 1] - g.starts[i]) == n || return nothing
        for k in 1:nax
            sk, Lk = strides[k], shape[k]
            c = rem(div(p, sk), Lk)
            expected[2k - 1] = c == 0        ? p + (Lk - 1) * sk : p - sk
            expected[2k]     = c == Lk - 1   ? p - (Lk - 1) * sk : p + sk
        end
        fill!(used, false)
        for c in g.starts[i]:(g.starts[i + 1] - 1)
            q = Int(g.flat[c]) - 1
            hit = 0
            for j in 1:n
                if !used[j] && expected[j] == q
                    hit = j
                    break
                end
            end
            hit == 0 && return nothing       # a neighbour that is not a lattice bond
            used[hit] = true
        end
    end

    return HopStructure(s.diags, shape, g.value)
end
