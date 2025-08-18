
mutable struct LanczosFactorization{E,P} <: KrylovFactorization{E,P}
    krylovdim::Int # current Krylov dimension
    basis::OrthonormalBasis # basis of length k
    r:: AbstractVector{E}
    v_last:: AbstractVector{E}
    v_secondlast:: AbstractVector{E}
    βs::AbstractVector{E}
    αs::AbstractVector{E}
    pu::P
    
    function LanczosFactorization(maxdim::Int, basis::OrthonormalBasis, x0::AbstractVector{E}, pu::P) where {E, P<:ProcessingUnit}
        r  = similar(x0)
        v_last = pu.zeros(E, size(x0))
        v_secondlast = pu.zeros(E, size(x0))

        αs = zeros(E, maxdim)
        βs = zeros(E, maxdim-1)

        return new{E,P}(0, basis, r, v_last, v_secondlast, βs, αs, pu)
    end
end





# get factorization properties
@inline getoverlap(fact::LanczosFactorization) = fact.αs[fact.krylovdim]
@inline getnorm(fact::LanczosFactorization) = fact.βs[fact.krylovdim-1]

# add factorization properties 
@inline addnorm!(fact::LanczosFactorization, β::Real) = begin fact.βs[fact.krylovdim-1] = β; end


# other functions
@inline constructfactorizedmat(fact::LanczosFactorization) = begin isa(fact.pu, GPU) ? CuMatrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) : Matrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) end
@inline calc_norm_krylovvec!(fact::LanczosFactorization) = begin β = LinearAlgebra.norm(fact.r); fact.r ./= β; return β; end

@inline calcoverlap!(fact::LanczosFactorization) = begin fact.αs[fact.krylovdim] = dot(fact.v_last, fact.r) end

@inline tridiagonalization!(fact::LanczosFactorization) = begin

    β = norm(fact.r)
    fact.r ./= β
    fact.βs[fact.krylovdim] = β

    update!(fact, fact.r)
end



