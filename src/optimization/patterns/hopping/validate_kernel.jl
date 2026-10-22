# hopping — validation.
#
#   can_apply   — BEFORE building: is this operator a periodic hopping operator at all, and can
#                 this variant express it?
#   auto_score  — should this variant be CHOSEN at this block size?
#
# The split matters: `rows` can run on everything `packed` can, so `packed`'s extra restrictions
# are stated where they belong rather than excluding the pattern.

"""
    can_apply(::Val{:hopping}, s::MappingStructure) -> Bool

`true` when the off-diagonals of `s` are exactly the `2D` periodic nearest-neighbour bonds of a
rectangular lattice, all carrying one amplitude — i.e. `s` is the Anderson operator
`diag(ε) + t·Σ_k (T_k⁺ + T_k⁻)`. See [`_derive_hop`](@ref).
"""
can_apply(::Val{:hopping}, s::MappingStructure) = !isnothing(_derive_hop(s))

"""    why_not(::Val{:hopping}, s::MappingStructure) -> String"""
function why_not(::Val{:hopping}, s::MappingStructure)
    n_values(s) == 1 ||
        return "the $(n_values(s)) off-diagonal values are not one uniform hopping amplitude"
    is_regular(s) ||
        return "row lengths run $(s.lmin) … $(s.lmax); a periodic lattice gives every site 2D bonds"
    return "the off-diagonals are not the periodic nearest-neighbour bonds of a rectangular " *
           "lattice (every extent must be at least 3, and sites must be linearised x-fastest)"
end

"""
    _hop_packed_ok(::Type{T}, bs::Integer) -> Bool

Whether `packed`'s block-size preconditions hold: a real element type (SIMD.jl `Vec` carries no
`Complex`) and a `bs` that divides the pack width, so the one-site shift is a lane permutation.

This is the `bs`-dependent half only, which is why it takes no structure. The lattice half —
declining an odd x-extent — lives in the three-argument [`can_apply`](@ref), where `s` is in
hand; between them they cover every reason `packed` might not be the kernel to build.
"""
function _hop_packed_ok(::Type{T}, bs::Integer) where {T}
    T <: Real || return false
    W = _hop_lane_width(T)
    return bs <= W && rem(W, bs) == 0
end

"""
    can_apply(::Val{:hopping}, ::Val{:packed}, s::MappingStructure) -> Bool

`packed` has two requirements beyond the pattern's.

A **real** element type: it is built on SIMD.jl `Vec`, which does not carry `Complex`. Complex
Hamiltonians take `rows`, which is generic.

An **even x-extent**: a pack covers `P = W/bs` adjacent sites of one x-line, so `P` must divide
`L₁`, and `P` is a power of two — an odd `L₁` therefore forces `P = 1` at every block size. At
`P = 1` a pack is one site and the kernel computes exactly what `rows` computes, but with
`rows`'s restrictions (dense `Matrix`, real eltype) and none of its generality. Declining here
routes those lattices to `rows` instead of silently building a one-lane kernel.

The block-size precondition cannot be checked here (`bs` is not known yet); it is enforced by
[`auto_score`](@ref) for the automatic path and by `_build_hop_packed` for a pinned request.
"""
function can_apply(::Val{:hopping}, ::Val{:packed}, s::MappingStructure)
    eltype(s) <: Real || return false
    hs = _derive_hop(s)
    return !isnothing(hs) && iseven(hs.shape[1])
end

"""    why_not(::Val{:hopping}, ::Val{:packed}, s::MappingStructure) -> String"""
function why_not(::Val{:hopping}, ::Val{:packed}, s::MappingStructure)
    eltype(s) <: Real ||
        return "packed is built on SIMD Vec, which carries no Complex; use hopping/rows"
    hs = _derive_hop(s)
    (isnothing(hs) || iseven(hs.shape[1])) ||
        return "the x-extent L₁ = $(hs.shape[1]) is odd, so a pack could only cover one site; " *
               "use hopping/rows, which is that kernel without the extra restrictions"
    return why_not(Val(:hopping), s)
end

"""
    auto_score(::Val{:hopping}, ::Val{:packed}, bs::Int) -> Int

Preferred over every other kernel whenever its block-size preconditions hold — it reads no index
table AND keeps the axis-1 neighbours in registers, which is ~2.7x `hopping/rows` at `bs = 1` on
the 3-D benchmark. Scores negative for a `bs` it cannot express, which routes the automatic path
to `rows`.
"""
auto_score(::Val{:hopping}, ::Val{:packed}, bs::Int) =
    _hop_packed_ok(Float64, bs) ? 100 : -1

"""
    auto_score(::Val{:hopping}, ::Val{:rows}, bs::Int) -> Int

Always eligible, and preferred over the `universal` kernels (it reads no index table), but below
`packed`.
"""
auto_score(::Val{:hopping}, ::Val{:rows}, ::Int) = 50
