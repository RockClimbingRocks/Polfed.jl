
const permutation = SVector{3,Int64}(3,1,2)


@inline function clenshaw_algorithm!(
    coefficients::Function, 
    order::Int, 
    α::Function, 
    β::Function, 
    ϕ₀::Real, 
    ϕ₁::Real, 
    mapping!::Function, 
    b::Vector{<:AbstractVecOrMat{<:Number}},
    X::AbstractVecOrMat{<:Number}, 
    Y::AbstractVecOrMat{<:Number}
)
    
    @inbounds @fastmath for k in order:-1:1
        reverse_recurrence_relation!(mapping!, b, coefficients(k), α(k), β(k+1), X)
        permute!(b, permutation);
    end
    
    finalsum!(coefficients(0), β(0+1), ϕ₀, ϕ₁, b, mapping!, X, Y)
        
end


@inline function clenshaw_algorithm!(   
    coefficients::Function, 
    order::Int, 
    recurrencerelation_kernel!::Function, 
    finalsum_kernel!::Function, 
    b::Vector{<:AbstractVecOrMat{<:Number}}, 
    X::AbstractVecOrMat{<:Number}, 
    Y::AbstractVecOrMat{<:Number}
)

    @inbounds @fastmath for k in order:-1:1
        CUDA.@sync begin
            recurrencerelation_kernel!(b[1], b[2], b[3], coefficients(k), X)
        end

        permute!(b, permutation);
    end

    CUDA.@sync begin
        finalsum_kernel!(b[2], b[3], coefficients(0), Y, X)
    end 
end





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
