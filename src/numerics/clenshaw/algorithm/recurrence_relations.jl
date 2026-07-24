"""
    mapping_recurrence!(mapping!, b1, b2, b3, c, α, β, X) -> b1

Apply one in-place reverse Clenshaw recurrence step for a vector/matrix state.

The recurrence advances the rotating buffers by computing

    b1 ← c·X + α·(A·b2) + β·b3

where the linear action `A` is supplied through the mutating callback
`mapping!(out, in)` (`out ← A·in`). For Chebyshev polynomials `α = 2`,
`β = -1`, reproducing `bₖ = cₖ·X + 2·A·bₖ₊₁ − bₖ₊₂`.

# Arguments
- `mapping!::Function`: in-place linear action, `mapping!(b1, b2)` writes `A·b2` into `b1`.
- `b1, b2, b3`: rotating recurrence buffers; `b1` is overwritten with the new state.
- `c::Number`: the current expansion coefficient `cₖ` (real or complex).
- `α, β::Real`: recurrence weights for the polynomial family (`αₖ`, `βₖ₊₁`).
- `X`: the seed term — an `AbstractVecOrMat` or a `UniformScaling`.

# Returns
- `b1`, mutated in place.

# Notes
The `UniformScaling` method exists so the *whole* expansion `S(A)` can be formed
(seed `X = I`) rather than only its action `S(A)·v` on a vector.
"""
@inline function mapping_recurrence!(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    mapping!(b1, b2)
    @inbounds @. b1 = c * X + α * b1 + β * b3
    return b1
end

"""`UniformScaling` seed overload of [`mapping_recurrence!`](@ref)."""
@inline function mapping_recurrence!(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::UniformScaling,
)
    mapping!(b1, b2)
    @inbounds @. b1 *= α
    b1 .+= c * X + β * b3
    return b1
end

"""
    scalar_recurrence(c, α, β, x, b2, b3) -> Number

Scalar form of one reverse Clenshaw recurrence step, returning the new buffer
value `bₖ = c + α·x·b2 + β·b3`.

This is the scalar analogue of [`mapping_recurrence!`](@ref); the linear action
`A` collapses to multiplication by the scalar argument `x`.
"""
@inline scalar_recurrence(c::Number, α::Real, β::Real, x::Number, b2::Number, b3::Number) =
    c + α * x * b2 + β * b3
