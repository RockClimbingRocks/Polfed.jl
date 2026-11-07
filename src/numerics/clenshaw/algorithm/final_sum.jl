"""
    mapping_finalsum!(mapping!, b2, b3, c₀, β₁, ϕ₀, ϕ₁, Y, X) -> Y

Apply the terminal Clenshaw combination for a vector/matrix state, writing the
fully evaluated expansion into `Y`:

    Y ← c₀·ϕ₀·X + ϕ₁·(A·b2) + β₁·ϕ₀·b3

where the linear action `A` is supplied through the mutating callback
`mapping!(out, in)` (`out ← A·in`).

# Arguments
- `mapping!::Function`: in-place linear action, `mapping!(Y, b2)` writes `A·b2` into `Y`.
- `b2, b3`: the two surviving recurrence buffers after the loop.
- `c₀::Number`: the zeroth expansion coefficient (real or complex).
- `β₁::Real`: recurrence weight `β(1)` for the family.
- `ϕ₀, ϕ₁::Real`: normalization constants of the polynomial family (`φ₀`, `φ₁` scale factors).
- `Y`: output buffer, overwritten with the result.
- `X`: the seed term — an `AbstractVecOrMat` or a `UniformScaling`.

# Returns
- `Y`, mutated in place.
"""
@inline function mapping_finalsum!(
    mapping!::Function,
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c₀::Number,
    β₁::Real,
    ϕ₀::Real,
    ϕ₁::Real,
    Y::AbstractVecOrMat{<:Number},
    X::AbstractVecOrMat{<:Number},
)
    mapping!(Y, b2)
    @inbounds @. Y = c₀ * ϕ₀ * X + ϕ₁ * Y + β₁ * ϕ₀ * b3
    return Y
end

"""`UniformScaling` seed overload of [`mapping_finalsum!`](@ref)."""
@inline function mapping_finalsum!(
    mapping!::Function,
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c₀::Number,
    β₁::Real,
    ϕ₀::Real,
    ϕ₁::Real,
    Y::AbstractVecOrMat{<:Number},
    X::UniformScaling,
)
    mapping!(Y, b2)
    @inbounds @. Y *= ϕ₁
    Y .+= c₀ * ϕ₀ * X + β₁ * ϕ₀ * b3
    return Y
end

"""
    scalar_finalsum(c₀, β₁, ϕ₀, ϕ₁, x, b2, b3) -> Number

Scalar form of the terminal Clenshaw combination, returning the evaluated
polynomial value `c₀·ϕ₀ + ϕ₁·x·b2 + β₁·ϕ₀·b3`.

This is the scalar analogue of [`mapping_finalsum!`](@ref).
"""
@inline scalar_finalsum(c₀::Number, β₁::Real, ϕ₀::Real, ϕ₁::Real, x::Number, b2::Number, b3::Number) =
    c₀ * ϕ₀ + ϕ₁ * x * b2 + β₁ * ϕ₀ * b3
