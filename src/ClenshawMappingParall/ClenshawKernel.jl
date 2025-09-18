
struct ClenshawKernel
    type:: Symbol
    coefficients:: Function
    order:: Int
    recurrencerelation_kernel!::Function
    finalsum_kernel!::Function
    D::Int
    T::Type

    function ClenshawKernel(coefficients::Union{Function,AbstractVector{<:Number}}, order::Int, type::Symbol, recurrencerelation_kernel!::Function, finalsum_kernel!::Function, D::Int, T::Type)

        coefficientsfun(i::Int) = isa(coefficients, Function) ? coefficients(i) : coefficients[i]
        new(type, coefficientsfun, order, recurrencerelation_kernel!, finalsum_kernel!, D, T)
    end
end




function (clenshaw::ClenshawKernel)(Y::AbstractVecOrMat{<:Real}, X::AbstractVecOrMat{<:Real}, b::Vector{<:AbstractVecOrMat{<:Real}})
    @assert length(b) == 3 "Vector b does not have length equal 3!"
    b[1] .= 0; b[2] .= 0; b[3] .= 0

    clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, clenshaw.recurrencerelation_kernel!, clenshaw.finalsum_kernel!, b, X, Y)
end