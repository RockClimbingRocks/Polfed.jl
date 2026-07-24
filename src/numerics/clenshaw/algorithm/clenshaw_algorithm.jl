"""
Rotation applied to the 3-buffer workspace after each recurrence step.

`permute!(b, permutation)` cycles `(b1, b2, b3) → (b3, b1, b2)`, so the freshly
written `b1` becomes `b2` (the most recent state) and the oldest buffer is reused
as scratch on the next iteration. Using a `StaticArrays.SVector` keeps the
permutation allocation-free.
"""
const permutation = SVector{3,Int64}(3, 1, 2)

"""
    clenshaw_algorithm!(coefficients, order, recurrence!, finalsum!, b, X, Y) -> Y

Run the matrix/vector Clenshaw recurrence in place.

Iterates the reverse recurrence from `k = order` down to `1` using the
`recurrence!` callback, rotating the workspace `b` after each step, then applies
`finalsum!` to obtain the result in `Y`.

# Arguments
- `coefficients::Function`: `coefficients(k)` returns the `k`-th expansion coefficient.
- `order::Int`: highest polynomial degree in the expansion.
- `recurrence!::Function`: step callback `(b1, b2, b3, c, k, X)`; see [`mapping_recurrence!`](@ref).
- `finalsum!::Function`: terminal callback `(b2, b3, c₀, Y, X)`; see [`mapping_finalsum!`](@ref).
- `b::Vector{<:AbstractVecOrMat}`: 3-buffer rotating workspace (assumed pre-zeroed).
- `X`: seed term (`AbstractVecOrMat` or `UniformScaling`).
- `Y`: output buffer, overwritten with the result.

# Returns
- `Y`, mutated in place.

# Notes
With `order == 0` the loop body never executes and the result is `finalsum!`
applied to the zero buffers — i.e. the constant term alone.
"""
@inline function clenshaw_algorithm!(
    coefficients::Function,
    order::Int,
    recurrence!::Function,
    finalsum!::Function,
    b::Vector{<:AbstractVecOrMat{<:Number}},
    X,
    Y::AbstractVecOrMat{<:Number},
)
    @inbounds for k in order:-1:1
        recurrence!(b[1], b[2], b[3], coefficients(k), k, X)
        permute!(b, permutation)
    end

    finalsum!(b[2], b[3], coefficients(0), Y, X)
    return Y
end

"""
    clenshaw_algorithm!(coefficients, order, α, β, ϕ₀, ϕ₁, b, x) -> Number

Run the scalar Clenshaw recurrence and return the polynomial value at `x`.

This is the scalar counterpart of the matrix/vector method: instead of mapping
callbacks it takes the polynomial family's recurrence weight functions `α`, `β`
and normalization constants `ϕ₀`, `ϕ₁` directly.

# Arguments
- `coefficients::Function`: `coefficients(k)` returns the `k`-th expansion coefficient.
- `order::Int`: highest polynomial degree in the expansion.
- `α, β::Function`: recurrence weight functions of the polynomial family.
- `ϕ₀, ϕ₁::Real`: normalization constants of the family.
- `b::AbstractVector{<:Number}`: length-3 scalar workspace (assumed pre-zeroed).
- `x::Number`: evaluation point.

# Returns
- The evaluated polynomial value at `x`.
"""
@inline function clenshaw_algorithm!(
    coefficients::Function,
    order::Int,
    α::Function,
    β::Function,
    ϕ₀::Real,
    ϕ₁::Real,
    b::AbstractVector{<:Number},
    x::Number,
)
    @inbounds @fastmath for k in order:-1:1
        b[1] = scalar_recurrence(coefficients(k), α(k), β(k + 1), x, b[2], b[3])
        permute!(b, permutation)
    end

    return scalar_finalsum(coefficients(0), β(1), ϕ₀, ϕ₁, x, b[2], b[3])
end
