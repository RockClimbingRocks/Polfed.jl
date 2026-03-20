
"""
    orthogonalization!(factorization::KrylovFactorization) -> nothing

Orthogonalize residual workspace against latest basis vectors/blocks using
current overlap and norm coefficients.
"""
@inline function orthogonalization!(factorization::KrylovFactorization)
    v1 = factorization.v_last
    v2 = factorization.v_secondlast

    α = getoverlap(factorization)
    β = getnorm(factorization)


    orthogonalization!(factorization.r, v1, v2, α, β)
end



"""
    orthogonalization!(r, v1, v2, α, β) -> nothing

Low-level orthogonalization kernels for vector and block variants. `r` is
mutated in-place.
"""
@inline function orthogonalization!(r::AbstractVector, v1::AbstractVector, v2::AbstractVector, α::Real, β::Real)
    @. r -= (α*v1 + β*v2)
end

@inline function orthogonalization!(r::AbstractMatrix, v1::AbstractMatrix, v2::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    r .-= (v1*α + v2*β')
end

"""Mixed vector/block overload of `orthogonalization!`."""
@inline function orthogonalization!(r::AbstractMatrix, v1::AbstractVector, v2::AbstractVector, α::AbstractMatrix, β::AbstractMatrix)
    r .-= (v1*α + v2*β')
end
