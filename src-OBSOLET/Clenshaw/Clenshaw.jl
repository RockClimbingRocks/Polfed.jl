
"""
    Clenshaw

Callable object implementing polynomial evaluation via the Clenshaw recurrence.

Constructors accept either:
- a mapping callback `mapping!(Ỹ, Y)`,
- explicit recurrence/final-sum callbacks, or
- a concrete matrix.
"""
struct Clenshaw
    type:: Symbol
    coefficients:: Function
    order:: Int
    D::Int
    T::Type
    recurrence!:: Function
    finalsum!:: Function

    """Build `Clenshaw` from mapping callback and polynomial metadata."""
    function Clenshaw(
        type::Symbol,
        coefficients::Union{Function,AbstractVector{<:Number}},
        order::Int,
        mapping!::Function,
        D::Int,
        T::Type,
    )
        coefficientsfun(i::Int) = isa(coefficients, Function) ? coefficients(i) : coefficients[i+1]
        recurrence!, finalsum! = build_clenshaw_kernels(mapping!, type)
        new(type, coefficientsfun, order, D, T, recurrence!, finalsum!)
    end

    """Build `Clenshaw` from explicit recurrence and final-sum callbacks."""
    function Clenshaw(
        type::Symbol,
        coefficients::Union{Function,AbstractVector{<:Number}},
        order::Int,
        recurrence!::Function,
        finalsum!::Function,
        D::Int,
        T::Type,
    )
        coefficientsfun(i::Int) = isa(coefficients, Function) ? coefficients(i) : coefficients[i+1]
        new(type, coefficientsfun, order, D, T, recurrence!, finalsum!)
    end

    """Build `Clenshaw` from explicit matrix mapping `Ỹ = mat * Y`."""
    function Clenshaw(
        type::Symbol,
        coefficients::Union{Function,AbstractVector{<:Number}},
        order::Int,
        mat::AbstractMatrix{<:Number}
    )
        coefficientsfun(i::Int) = isa(coefficients, Function) ? coefficients(i) : coefficients[i+1]
        mapping!(Ỹ::AbstractVecOrMat{<:Number}, Y::AbstractVecOrMat{<:Number}) = mul!(Ỹ, mat, Y)
        recurrence!, finalsum! = build_clenshaw_kernels(mapping!, type)
        new(type, coefficientsfun, order, size(mat,1), eltype(mat), recurrence!, finalsum!)
    end
end


"""
    mapping_recurrence!(...)

Generic mapping-backed recurrence update used by Clenshaw kernels.
"""
@inline function mapping_recurrence!(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Real,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    mapping!(b1, b2)
    @inbounds @. b1 = c*X + α*b1 + β*b3
end

"""`UniformScaling` overload of `mapping_recurrence!`."""
@inline function mapping_recurrence!(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Real,
    α::Real,
    β::Real,
    X::UniformScaling,
)
    mapping!(b1, b2)
    @inbounds @. b1 *= α
    b1 .+= c*X + β*b3
end

"""
    mapping_finalsum!(...)

Generic mapping-backed final accumulation used by Clenshaw kernels.
"""
@inline function mapping_finalsum!(
    mapping!::Function,
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c₀::Real,
    β₁::Real,
    ϕ₀::Real,
    ϕ₁::Real,
    Y::AbstractVecOrMat{<:Number},
    X::AbstractVecOrMat{<:Number},
)
    mapping!(Y, b2)
    @inbounds @. Y = c₀*ϕ₀*X + ϕ₁*Y + β₁*ϕ₀*b3
end

"""`UniformScaling` overload of `mapping_finalsum!`."""
@inline function mapping_finalsum!(
    mapping!::Function,
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c₀::Real,
    β₁::Real,
    ϕ₀::Real,
    ϕ₁::Real,
    Y::AbstractVecOrMat{<:Number},
    X::UniformScaling,
)
    mapping!(Y, b2)
    @inbounds @. Y *= ϕ₁
    Y .+= c₀*ϕ₀*X + β₁*ϕ₀*b3
end

"""
    build_clenshaw_kernels(mapping!::Function, type::Symbol) -> Tuple{Function,Function}

Create recurrence and final-sum callbacks from polynomial properties for `type`.
"""
@inline function build_clenshaw_kernels(mapping!::Function, type::Symbol)
    α, β, ϕ₀, ϕ₁ = polynomial_properties[type]

    recurrence! = (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, k::Int, X) -> begin
        mapping_recurrence!(mapping!, b1, b2, b3, c, α(k), β(k+1), X)
    end

    finalsum! = (b2::AbstractVecOrMat, b3::AbstractVecOrMat, c₀::Real, Y::AbstractVecOrMat, X) -> begin
        mapping_finalsum!(mapping!, b2, b3, c₀, β(1), ϕ₀, ϕ₁, Y, X)
    end

    return recurrence!, finalsum!
end


"""
    (clenshaw::Clenshaw)(Y, X, b) -> nothing

Apply the Clenshaw transform to vector/matrix input `X`, writing result to `Y`.
`b` is a mutable 3-buffer workspace.
"""
function (clenshaw::Clenshaw)(Y::AbstractVecOrMat{<:Number}, X::AbstractVecOrMat{<:Number}, b::Vector{<:AbstractVecOrMat{<:Number}})
    b[1] .= 0; b[2] .= 0; b[3] .= 0
    @assert length(b) == 3 "Vector b does not have length equal 3!"

    clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, clenshaw.recurrence!, clenshaw.finalsum!, b, X, Y)
end

"""
    (clenshaw::Clenshaw)(x::Real) -> Real

Evaluate the scalar polynomial represented by `clenshaw` at `x`.
"""
function (clenshaw::Clenshaw)(x::Real)
    b = zeros(typeof(x), 3)

    α, β, ϕ₀, ϕ₁ = polynomial_properties[clenshaw.type]
    y = clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, α, β, ϕ₀, ϕ₁, b, x)
    return y
end
