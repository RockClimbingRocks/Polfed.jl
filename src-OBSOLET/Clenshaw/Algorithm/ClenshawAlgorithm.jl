
const permutation = SVector{3,Int64}(3,1,2)

"""
    clenshaw_algorithm!(coefficients::Function, order::Int, recurrence!::Function, finalsum!::Function, b, X, Y) -> nothing

Execute matrix/vector Clenshaw recurrence in-place.

`b` is a rotating 3-buffer workspace; `Y` is mutated with final result.
"""

@inline function clenshaw_algorithm!(   
    coefficients::Function, 
    order::Int, 
    recurrence!::Function, 
    finalsum!::Function, 
    b::Vector{<:AbstractVecOrMat{<:Number}}, 
    X::AbstractVecOrMat{<:Number}, 
    Y::AbstractVecOrMat{<:Number}
)

    @inbounds for k in order:-1:1

        recurrence!(b[1], b[2], b[3], coefficients(k), k, X)
        permute!(b, permutation);
    end

    finalsum!(b[2], b[3], coefficients(0), Y, X)
end
"""
    clenshaw_algorithm!(coefficients::Function, order::Int, α::Function, β::Function, ϕ₀::Real, ϕ₁::Real, b::Vector{<:Real}, x::Real) -> Real

Execute scalar Clenshaw recurrence and return polynomial value at `x`.
"""
@inline function clenshaw_algorithm!(
    coefficients::Function, 
    order::Int, 
    α::Function, 
    β::Function, 
    ϕ₀::Real, 
    ϕ₁::Real, 
    b::Vector{<:Real}, 
    x::Real
)
    
    @inbounds @fastmath for k in order:-1:1
        # reverse_recurrence_relation!(mapping!, b, coefficients(k), α(k), β(k+1), X)
        reverse_recurrence_relation!(x, b, coefficients(k), α(k), β(k+1))
        permute!(b, permutation);
    end
    
    y = finalsum!(coefficients(0), β(0+1), ϕ₀, ϕ₁, b, x)
    return y
end
