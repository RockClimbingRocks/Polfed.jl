"""
    _threaded_combine!(b1, X, b3, c, α, β) -> b1

`b1 ← c·X + α·b1 + β·b3`, elementwise and chunked over threads.

Chunk-based, never indexed by `threadid()`: tasks migrate on Julia >= 1.12, so a per-thread
index is not bounded by the thread count. The single-thread branch skips the parallel region
entirely.
"""
function _threaded_combine!(b1, X, b3, c, α, β)
    n = length(b1)
    n == 0 && return b1

    nt = min(Base.Threads.nthreads(:default), n)
    chunk = cld(n, nt)

    if nt == 1
        @inbounds @simd for i in eachindex(b1)
            b1[i] = c * X[i] + α * b1[i] + β * b3[i]
        end
    else
        Base.Threads.@threads :static for t in 1:nt
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, n)

            @inbounds @simd for i in lo:hi
                b1[i] = c * X[i] + α * b1[i] + β * b3[i]
            end
        end
    end

    return b1
end

"""
    _threaded_combine_scaling!(b1, λ, b3, c, α, β) -> b1

`b1 ← c·λ·I + α·b1 + β·b3` — the `X::UniformScaling` case of [`_threaded_combine!`](@ref).

A `UniformScaling` has no element to load at index `i`, which is why it cannot share the
elementwise loop: `c·λ·I` touches the DIAGONAL only. So the two contributions are two different
loops — a full elementwise pass for `α·b1 + β·b3`, then `n` diagonal updates.

Writing it as `b1 .= c*(λ*I) .+ α.*b1 .+ β.*b3` would not work either (`UniformScaling` does not
broadcast), and the obvious repair `b1 .+= c*λ*I + β*b3` materialises a dense `D×D` temporary on
EVERY recurrence step — the whole matrix the expansion is being built to produce, allocated
`order` times over.

# Throws
- `DimensionMismatch` if `b1` is not square, matching `LinearAlgebra`'s own rule for adding a
  `UniformScaling` to a matrix: there is no unique diagonal otherwise.
"""
function _threaded_combine_scaling!(b1::AbstractMatrix, λ, b3, c, α, β)
    m, n = size(b1)
    m == n || throw(DimensionMismatch(
        "mapping_recurrence!: a UniformScaling seed adds c·λ to the diagonal, which needs a " *
        "square buffer; got $(m)×$(n)."))
    N = length(b1)
    N == 0 && return b1

    nt = min(Base.Threads.nthreads(:default), N)
    chunk = cld(N, nt)

    if nt == 1
        @inbounds @simd for i in eachindex(b1)
            b1[i] = α * b1[i] + β * b3[i]
        end
    else
        Base.Threads.@threads :static for t in 1:nt
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, N)

            @inbounds @simd for i in lo:hi
                b1[i] = α * b1[i] + β * b3[i]
            end
        end
    end

    cλ = c * λ
    @inbounds for k in 1:n
        b1[k, k] += cλ
    end

    return b1
end

"""
Vector buffers with a `UniformScaling` seed: refused rather than guessed at.

The seed exists so the WHOLE expansion `S(A)` can be formed, which is a matrix; `c·I` has no
meaning added to a vector state, and `LinearAlgebra` refuses the same combination.
"""
_threaded_combine_scaling!(b1::AbstractVector, λ, b3, c, α, β) = throw(ArgumentError(
    "mapping_recurrence!: a UniformScaling seed forms the whole expansion S(A) and needs matrix " *
    "buffers; got a $(length(b1))-element vector. Pass an explicit seed vector instead."))



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
    # print("mapping:   ")
    # @time mapping!(b1, b2)
    # print("broadcast: ")
    # @time _threaded_combine!(b1, X, b3, c, α, β)
    mapping!(b1, b2)
    _threaded_combine!(b1, X, b3, c, α, β)
    return b1
end



@inline function mapping_recurrence!2(
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
    _threaded_combine!(b1, X, b3, c, α, β)
    b3,b2 = b2,b1
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
    _threaded_combine_scaling!(b1, X.λ, b3, c, α, β)
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
