
struct Clenshaw
    type:: Symbol
    coefficients:: Function
    order:: Int
    mapping!:: Function
    D::Int
    T::Type


    function Clenshaw(type::Symbol, coefficients::Union{Function,AbstractVector{<:Number}}, order::Int, mapping!::Function, D::Int, T::Type)

        coefficientsfun(i::Int) = isa(coefficients, Function) ? coefficients(i) : coefficients[i+1]
        new(type, coefficientsfun, order, mapping!, D, T)
    end


    function Clenshaw(type::Symbol, coefficients::Union{Function,AbstractVector{<:Number}}, order::Int, mat::AbstractMatrix{<:Real})

        coefficientsfun(i::Int) = isa(coefficients, Function) ? coefficients(i) : coefficients[i]
        mapping!(Ỹ::AbstractVecOrMat{<:Number}, Y::AbstractVecOrMat{<:Number}) = mul!(Ỹ, mat, Y)
        new(type, coefficientsfun, order, mapping!, size(mat,1), eltype(mat))
    end
end


function (clenshaw::Clenshaw)(Y::AbstractVecOrMat{<:Real}, X::AbstractVecOrMat{<:Real}, b::Vector{<:AbstractVecOrMat{<:Real}})
    b[1] .= 0; b[2] .= 0; b[3] .= 0
    @assert length(b) == 3 "Vector b does not have length equal 3!"

    α, β, ϕ₀, ϕ₁ = polynomial_properties[clenshaw.type]
    clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, α, β, ϕ₀, ϕ₁, clenshaw.mapping!, b, X, Y)
end

function (clenshaw::Clenshaw)(x::Real)
    b = zeros(typeof(x), 3)
    
    α, β, ϕ₀, ϕ₁ = polynomial_properties[clenshaw.type]
    y = clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, α, β, ϕ₀, ϕ₁, b, x)
    return y
end 

