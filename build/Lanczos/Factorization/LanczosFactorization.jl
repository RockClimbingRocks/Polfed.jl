
"""
    LanczosFactorization(maxdim::Int, basis::OrthonormalBasis, x0::AbstractVector{T}, pu::ProcessingUnit)

State container for scalar (non-block) Lanczos factorization.
"""
mutable struct LanczosFactorization{S<:Real,T<:Number,P} <: KrylovFactorization{S,T,P}
    krylovdim::Int # current Krylov dimension
    basis::OrthonormalBasis # basis of length k
    r:: AbstractVector{T}
    v_last:: AbstractVector{T}
    v_secondlast:: AbstractVector{T}
    βs::AbstractVector{S}
    αs::AbstractVector{S}
    pu::P
    
    """Build initial scalar-Lanczos factorization state."""
    function LanczosFactorization(maxdim::Int, basis::OrthonormalBasis, x0::AbstractVector{T}, pu::P) where {T<:Number, P<:ProcessingUnit}
        S = real(T)
        r  = similar(x0)
        v_last = pu.zeros(T, size(x0))
        v_secondlast = pu.zeros(T, size(x0))

        αs = zeros(S, maxdim)
        βs = zeros(S, maxdim-1)

        return new{S,T,P}(0, basis, r, v_last, v_secondlast, βs, αs, pu)
    end
end





# get factorization properties
"""
    getoverlap, getnorm, addnorm!, constructfactorizedmat, calc_norm_krylovvec!, calcoverlap!, tridiagonalization!

Lanczos-factorization-specific helpers for overlap/norm bookkeeping, projected
matrix construction, and tridiagonalization updates.
"""
@inline getoverlap(fact::LanczosFactorization) = fact.αs[fact.krylovdim]
"""Return current scalar norm-coupling value `β`."""
@inline getnorm(fact::LanczosFactorization) = fact.βs[fact.krylovdim-1]

# add factorization properties 
"""Store scalar norm-coupling value `β` for current iteration."""
@inline addnorm!(fact::LanczosFactorization, β::Real) = begin fact.βs[fact.krylovdim-1] = β; end


# other functions
"""Construct projected tridiagonal matrix from stored Lanczos coefficients."""
@inline constructfactorizedmat(fact::LanczosFactorization) = begin isa(fact.pu, GPU) ? CuMatrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) : Matrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) end
"""Normalize residual vector and return its norm."""
@inline calc_norm_krylovvec!(fact::LanczosFactorization) = begin β = LinearAlgebra.norm(fact.r); fact.r ./= β; return β; end

"""Compute and store scalar overlap `α_k` for current iteration."""
@inline calcoverlap!(fact::LanczosFactorization) = begin fact.αs[fact.krylovdim] = real(dot(fact.v_last, fact.r)) end

"""Advance scalar-Lanczos tridiagonalization by one iteration."""
@inline tridiagonalization!(fact::LanczosFactorization) = begin

    β = norm(fact.r)
    fact.r ./= β
    fact.βs[fact.krylovdim] = β

    update!(fact, fact.r)
end
