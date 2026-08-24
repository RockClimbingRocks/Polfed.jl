# spin_flip — validation.
#
#   can_apply   — BEFORE building: is this operator a spin-flip operator at all, and can this
#                 variant express it?
#   auto_score  — should this variant be CHOSEN at this block size?
#
# The split matters: `rows` can run on everything `packed` can, so `packed`'s extra
# restrictions are stated where they belong rather than excluding the pattern.

"""
    can_apply(::Val{:spin_flip}, s::MappingStructure) -> Bool

`true` when the off-diagonals of `s` are exactly a fixed set of xor masks — i.e. `s` is
`diag(ε) + Σ_k v_k · X^{mask_k}` on a full `2^L` space. See [`_derive_flip`](@ref).
"""
can_apply(::Val{:spin_flip}, s::MappingStructure) = !isnothing(_derive_flip(s))

"""    why_not(::Val{:spin_flip}, s::MappingStructure) -> String"""
function why_not(::Val{:spin_flip}, s::MappingStructure)
    ispow2(s.dim) ||
        return "dim = $(s.dim) is not 2^L, so the row index is not a spin configuration"
    return "the off-diagonals are not a fixed set of xor masks (neighbour ≠ xor(row, mask))"
end

"""
    _packed_ok(::Type{T}, bs::Integer, dim::Integer) -> Bool

Whether `packed`'s correctness preconditions hold: a real element type (SIMD.jl `Vec` carries
no `Complex`), a power-of-two `bs` no wider than the pack, and a space at least one pack wide.
"""
function _packed_ok(::Type{T}, bs::Integer, dim::Integer) where {T}
    T <: Real || return false
    ispow2(bs) || return false
    W = _packed_lane_width(T)
    bs <= W || return false
    return dim >= W ÷ bs
end

"""
    can_apply(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure) -> Bool

`packed` additionally needs a **real** element type: it is built on SIMD.jl `Vec`, which does
not carry `Complex`. Complex Hamiltonians take `rows`, which is generic.

The block-size preconditions cannot be checked here (`bs` is not known yet); they are enforced
by [`auto_score`](@ref) for the automatic path and by `_build_packed` for a pinned request.
"""
can_apply(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure) =
    eltype(s) <: Real && can_apply(Val(:spin_flip), s)

"""    why_not(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure) -> String"""
function why_not(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure)
    eltype(s) <: Real ||
        return "packed is built on SIMD Vec, which carries no Complex; use spin_flip/rows"
    return why_not(Val(:spin_flip), s)
end

"""
    auto_score(::Val{:spin_flip}, ::Val{:packed}, bs::Int) -> Int

Preferred over every other kernel whenever its block-size preconditions hold — on QSun at L=20
it is ~45x `universal/row_read` at bs=1. Scores negative for a `bs` it cannot express, which
routes the automatic path to `rows`.
"""
auto_score(::Val{:spin_flip}, ::Val{:packed}, bs::Int) =
    (ispow2(bs) && bs <= _packed_lane_width(Float64)) ? 100 : -1

"""
    auto_score(::Val{:spin_flip}, ::Val{:rows}, bs::Int) -> Int

Always eligible, and preferred over the `universal` kernels (it reads no index table), but
below `packed`.
"""
auto_score(::Val{:spin_flip}, ::Val{:rows}, ::Int) = 50
