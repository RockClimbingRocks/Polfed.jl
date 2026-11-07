# FlipStructure — what a spin-flip operator IS, in the only form its kernels need.
#
#   H = diag(ε) + Σ_k v_k · X^{mask_k}          on the full space, D = 2^L
#   y[i] = ε[i]·x[i] + Σ_k v_k · x[xor(i-1, mask_k) + 1]
#
# `MappingStructure` (value-grouped CSR) is the wrong shape here: a flip operator has no
# neighbour tables, because the neighbour is ARITHMETIC. Storing one would cost D·M·4 bytes
# (1.6 GB at L=24, M=24) to hold information already implied by M integers.
#
# There are deliberately TWO ways to get one, and the second is the bigger win:
#
#   1. from a `MappingStructure` — `_derive_flip`, so the existing matrix entrypoints pick the
#      pattern up with no API change;
#   2. directly from `(diags, masks, values)` — no `H` is ever assembled. Materialising the
#      sparse matrix for QSun costs ~350 MB at L=20 and ~5.6 GB at L=24 purely to be thrown
#      away after `classify`.

"""
    FlipStructure{T}

A spin-flip operator: a diagonal plus `M` xor-mask terms.

# Fields
- `dim::Int`: `D = 2^L`.
- `L::Int`: number of spins, `log2(dim)`.
- `diags::Vector{T}`: the diagonal, length `D`.
- `masks::Vector{Int}`: 0-based xor masks. Normalised at construction — distinct, nonzero,
  all `< dim`, ascending.
- `values::Vector{T}`: the amplitude of every entry mask `k` generates, `values[k]`.

Rows are regular by construction: every row has exactly `length(masks)` off-diagonal entries.

See [`FlipStructure(diags, masks, values)`](@ref) for the normalisation rules.
"""
struct FlipStructure{T<:Number}
    dim::Int
    L::Int
    diags::Vector{T}
    masks::Vector{Int}
    values::Vector{T}
end

"""
    FlipStructure(diags, masks, values) -> FlipStructure

Build a spin-flip operator directly from its term list — **no matrix required**.

```julia
L = 20; D = 1 << L
fs = FlipStructure(randn(D), [1 << b for b in 0:L-1], fill(0.7, L))   # transverse field
```

# Normalisation
The mask list is cleaned up here rather than in the kernels, because each of these is a
silent wrong answer rather than an error at matvec time:

- a **zero mask** is not an off-diagonal at all — it is `x[i]` again, so it is folded into
  `diags`;
- **duplicate masks are merged** (their values summed); left alone they visit the same
  neighbour twice, which wastes a load and is right only by accident;
- a term whose merged value is exactly zero is dropped;
- masks are sorted ascending so the same operator always produces the same kernel.

# Throws
- `ArgumentError` if `dim` is not a power of two, if the lengths disagree, or if any mask is
  negative or `>= dim` — the kernels run under `@inbounds`, so an out-of-range mask would
  read garbage rather than throw.
"""
function FlipStructure(diags::Vector{T}, masks::AbstractVector{<:Integer},
                       values::AbstractVector) where {T<:Number}
    D = length(diags)
    (D > 0 && ispow2(D)) || throw(ArgumentError(
        "FlipStructure: dim must be a power of two (the full 2^L space), got $D."))
    length(masks) == length(values) || throw(ArgumentError(
        "FlipStructure: $(length(masks)) masks but $(length(values)) values."))
    L = trailing_zeros(D)

    acc = Dict{Int,T}()
    diag_extra = zero(T)
    for (m, v) in zip(masks, values)
        mi = Int(m)                       # never let an unsigned literal into the arithmetic
        (0 <= mi < D) || throw(ArgumentError(
            "FlipStructure: mask $mi outside 0:$(D - 1); xor(i-1, mask) would leave the array."))
        vT = convert(T, v)
        if mi == 0
            diag_extra += vT              # a zero mask is a diagonal contribution
        else
            acc[mi] = get(acc, mi, zero(T)) + vT
        end
    end

    ks = sort!(collect(keys(acc)))
    filter!(m -> !iszero(acc[m]), ks)
    dg = iszero(diag_extra) ? copy(diags) : diags .+ diag_extra
    return FlipStructure{T}(D, L, dg, ks, T[acc[m] for m in ks])
end

Base.eltype(::Type{FlipStructure{T}}) where {T} = T

"""    n_masks(fs::FlipStructure) -> Int — number of off-diagonal terms (= entries per row)."""
n_masks(fs::FlipStructure) = length(fs.masks)

"""
    rescale(fs::FlipStructure, a, b) -> FlipStructure

The structure of ``\\tilde H = (H - b)/a``: shift and scale the diagonal, scale every term
value. Masks are untouched — rescaling never moves an off-diagonal.
"""
function rescale(fs::FlipStructure{T}, a::Real, b::Real) where {T}
    inv_a = one(T) / T(a)
    return FlipStructure{T}(fs.dim, fs.L, @.((fs.diags - T(b)) * inv_a),
                            copy(fs.masks), fs.values .* inv_a)
end

"""
    _derive_flip(s::MappingStructure) -> FlipStructure or nothing

Recover the xor-mask term list from a value-grouped structure, or `nothing` if `s` is not a
spin-flip operator. This is what lets `mapping_kernels(H)` report the pattern and what lets
the existing matrix entrypoints use it with no API change.

Row 1 is state `p = 0`, so its neighbour columns minus one ARE the candidate masks. Every
other row then has to agree: row `i`'s neighbours in a group must be exactly
`{xor(i-1, m) + 1}` over that group's masks. The check is one pass over the nonzeros with a
dictionary lookup each — `O(nnz)`, the same order as `classify` itself.

Returns `nothing` (rather than throwing) because this is the question `can_apply` asks.
"""
function _derive_flip(s::MappingStructure{T,TI}) where {T,TI}
    D = s.dim
    (D > 0 && ispow2(D)) || return nothing

    masks = Int[]
    values = T[]
    # mask → index into `masks`, for the per-row verification below
    index_of = Dict{Int,Int}()
    for g in s.groups
        lo, hi = g.starts[1], g.starts[2] - 1
        hi >= lo || return nothing                        # a group with no entry in row 1
        for c in lo:hi
            m = Int(g.flat[c]) - 1                        # xor(0, m) = m
            (0 < m < D) || return nothing
            haskey(index_of, m) && return nothing         # the same mask in two groups
            push!(masks, m); push!(values, g.value)
            index_of[m] = length(masks)
        end
    end
    isempty(masks) && return nothing

    # every row must generate exactly this mask set, once each
    seen = fill(0, length(masks))
    for i in 1:D
        n = 0
        for g in s.groups
            for c in g.starts[i]:(g.starts[i + 1] - 1)
                m = xor(i - 1, Int(g.flat[c]) - 1)
                k = get(index_of, m, 0)
                k == 0 && return nothing                  # neighbour is not a known flip
                values[k] == g.value || return nothing    # same mask, different amplitude
                seen[k] == i && return nothing            # visited twice in this row
                seen[k] = i
                n += 1
            end
        end
        n == length(masks) || return nothing              # a row missing a term
    end

    # Sort ascending, values following their mask. `_derive_flip` bypasses the normalising
    # constructor (it has already done the validation itself), so it has to honour the same
    # invariant: the order masks happen to appear in row 1 is arbitrary, and letting it through
    # would make the SAME operator compile to two different `Val(MASKS)` kernels.
    perm = sortperm(masks)
    return FlipStructure{T}(D, trailing_zeros(D), s.diags, masks[perm], values[perm])
end
