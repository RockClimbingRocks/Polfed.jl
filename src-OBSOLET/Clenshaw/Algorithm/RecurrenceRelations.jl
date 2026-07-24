"""
    reverse_recurrence_relation!(...)

Apply one reverse Clenshaw recurrence update.

Methods support:
- matrix/vector mapping callbacks,
- `UniformScaling` source,
- scalar polynomial evaluation.
"""
@inline function reverse_recurrence_relation!(mapping!::Function, b::AbstractVector{<:AbstractVecOrMat{<:Number}}, cₖ::Real, αₖ::Real, βₖ₊₁::Real, X::AbstractVecOrMat{<:Number})
    # @inbounds @. b[1] *= αₖ
    # @. b[1] += cₖ*X + βₖ₊₁ * b[3]
    mapping!(b[1], b[2])
    @inbounds @. b[1] = cₖ*X + αₖ * b[1] + βₖ₊₁ * b[3]
end

"""`UniformScaling` overload of `reverse_recurrence_relation!`."""
@inline function reverse_recurrence_relation!(mapping!::Function, b::AbstractVector{<:AbstractVecOrMat{<:Number}}, cₖ::Real, αₖ::Real, βₖ₊₁::Real, X::UniformScaling)
    mapping!(b[1], b[2])
    @inbounds @. b[1] *= αₖ
    b[1] .+= cₖ*X + βₖ₊₁ * b[3]
end

"""Scalar-evaluation overload of `reverse_recurrence_relation!`."""
@inline function reverse_recurrence_relation!(x::Real, b::AbstractVector{<:Number}, cₖ::Real, αₖ::Real, βₖ₊₁::Real)
    b[1] = cₖ + αₖ * x * b[2] + βₖ₊₁ * b[3]
end


# Should be allright, i just need to figutre out how to pass \alpha and \beta paramethers efficiently 
# @inline function recurrencerelation_Jacobi!(X::AbstractMatrix{T}, c::Real, b::Vector{Matrix{T}}, i::Int64, Y::Union{AbstractMatrix{T}, UniformScaling}, MD::Symbol) where {T<:Number}
#     c1 = (2n+α+β+1)*(2n+α+β+2)/(2*(n+1)*(n+α+β+1))
#     c2 = -2*(n+α)*(n+β)*(2n+α+β+2)/(2*(n+1)*(n+α+β+1)*(2n+α+β))
#     c3 = (2n+α+β+3)(α^2-β^2)/(2*(n+2)*(n+α+β+2)*(2n+α+β+2))
#     MD == :LM ? mul!(b[2], b[1], X, c1, c2) : mul!(b[2], X, b[1], c1, c2)
#     @. b[2]+=c3*b[1]
#     Y==I ? b[2]+=c*I : @. b[2]+=c*Y
#     permute!(b, permutation);
# end


# Should be allright, i just need to figutre out how to pass \alpha and \beta paramethers efficiently 
# @inline function recurrencerelation_Laguerre!(X::AbstractMatrix{T}, c::Real, b::Vector{Matrix{T}}, i::Int64, Y::Union{AbstractMatrix{T}, UniformScaling}, MD::Symbol) where {T<:Number}
#     MD == :LM ? mul!(b[2], b[1], X, -1/(i+1), -(k+α+1)/(k+2)) : mul!(b[2], X, b[1], -1/(i+1), -(k+α+1)/(k+2))
#     @. b[2]+=(2i+α+1)/(i+1) *b[1]
#     Y==I ? b[2]+=c*I : @. b[2]+=c*Y
#     permute!(b, permutation);
# end
