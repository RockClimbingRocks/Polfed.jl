"""
    finalsum!(...)

Compute the final Clenshaw combination from recurrence buffers.

Methods support:
- matrix/vector mapping callbacks with explicit output buffer `Y`,
- scalar polynomial evaluation mode.
"""
function finalsum!(c₀::Real, β₁::Real, ϕ₀::Real, ϕ₁::Real, b::AbstractVector{<:AbstractVecOrMat{<:Number}}, mapping!::Function, X::AbstractVecOrMat{<:Number}, Y::AbstractVecOrMat{<:Number})
    mapping!(Y, b[2])
    @. @inbounds Y = c₀*ϕ₀*X + ϕ₁*Y + β₁*ϕ₀*b[3]
    # @inbounds @.  Y *= ϕ₁
    # @. Y += c₀*ϕ₀*X + β₁*ϕ₀*b[3]
end

"""`UniformScaling` overload of `finalsum!`."""
function finalsum!(c₀::Real, β₁::Real, ϕ₀::Real, ϕ₁::Real, b::AbstractVector{<:AbstractVecOrMat{<:Number}}, mapping!::Function, X::UniformScaling, Y::AbstractVecOrMat{<:Number})
    mapping!(Y, b[2])
    @inbounds @.  Y *= ϕ₁
    Y .+= c₀*ϕ₀*X + β₁*ϕ₀*b[3]
end


"""Scalar-evaluation overload of `finalsum!`."""
function finalsum!(c₀::Real, β₁::Real, ϕ₀::Real, ϕ₁::Real, b::AbstractVector{<:Number}, x::Real)
    y = c₀*ϕ₀ + ϕ₁*x*b[2] + β₁*ϕ₀*b[3]

    return y
end







# this is for other polynomials.... TO DO... 

# function finalsum_Jacobi!(X::AbstractMatrix{T}, c::Real, b::Vector{Matrix{T}}, X̃::AbstractMatrix{T}, Y::Union{AbstractMatrix{T},UniformScaling}, MD::Symbol) where {T<:Number}
#     MD==:LM ? mul!(X̃, b[1], X, 0.5(α+β-2), 0.) : mul!(X̃, X, b[1], 0.5(α+β-2), 0.)
#     X̃ .+= (c*I)*Y + (0.5(α-β)I)*b[1] + (-α*β*(α+β+2)/((α+β+1)*(α+β))*I)*b[2]
# end


# function finalsum_Laguerre!(X::AbstractMatrix{T}, c::Real, b::Vector{Matrix{T}}, X̃::AbstractMatrix{T}, Y::Union{AbstractMatrix{T},UniformScaling}, MD::Symbol) where {T<:Number}
#     MD==:LM ? mul!(X̃, b[1], X, -1, 0) : mul!(X̃, X, b[1], -1, 0)
#     X̃ .+= (c*I)*Y + ((α+1)I)*b[1] + (0.5(α+1)*I)*b[2]
# end
