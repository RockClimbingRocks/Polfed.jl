

"""
    BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, v0::AbstractMatrix{T}, pu::ProcessingUnit)

State container for block Lanczos factorization.
"""
mutable struct BlockLanczosFactorization{S<:Real,T<:Number,P} <: KrylovFactorization{S,T,P}
    krylovdim::Int # current Krylov dimension
    blocksize::Int # block dimension
    basis::OrthonormalBasis # basis of length k
    r::  AbstractMatrix{T}
    v_last:: AbstractMatrix{T}
    v_secondlast:: AbstractMatrix{T}
    mat::AbstractMatrix{T}
    pu::P
    
    """Build initial block-Lanczos factorization state."""
    function BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, v0::AbstractMatrix{T}, pu::P) where {T<:Number, P<:ProcessingUnit}
        S = real(T)
        r               = similar(v0)
        v_last          = pu.zeros(T, size(v0))
        v_secondlast    = pu.zeros(T, size(v0))
        mat = pu.zeros(T, maxdim, maxdim)

        return new{S,T,P}(0, size(v0, 2), basis, r, v_last, v_secondlast, mat, pu)
    end
    
end



# get factorization properties
"""
    getoverlap, getnorm, addnorm!, constructfactorizedmat, calc_norm_krylovvec!, calcoverlap!, tridiagonalization!

Block-Lanczos-specific helpers for overlap/norm bookkeeping, projected matrix
construction, and block tridiagonalization updates.
"""

function getoverlap(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    α = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim);
    return α
    
end

"""Return current block norm-coupling matrix `β` from block factorization state."""
function getnorm(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    β = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-2blocksize+1:krylovdim-blocksize);
    return β
    
end


# add factorization properties 

"""Store block norm-coupling matrix `β` into projected block-tridiagonal matrix."""
function addnorm!(fact::BlockLanczosFactorization, β::AbstractMatrix)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    β_  = view(fact.mat, krylovdim-blocksize+1:krylovdim   , krylovdim-2blocksize+1:krylovdim-blocksize);
    β′_ = view(fact.mat, krylovdim-2blocksize+1:krylovdim-blocksize, krylovdim-blocksize+1:krylovdim   );

    copyto!(β_, β)
    copyto!(β′_, β')    
end




# other functions

"""Return current projected factorized matrix view (Hermitian on CPU, CuMatrix on GPU)."""
function constructfactorizedmat(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim
    factmat = view(fact.mat, 1:krylovdim, 1:krylovdim)
    # display(factmat)

    return isa(fact.pu, GPU) ? CuMatrix(factmat) : Hermitian(factmat)
end

"""Normalize residual block via QR and return `R` factor."""
function calc_norm_krylovvec!(fact::BlockLanczosFactorization)
    # QR = qr!(factorization.r)
    QR = qr(fact.r)

    β = QR.R
    copy!(fact.r, fact.pu.Matrix(QR.Q))

    return β
end


"""Compute and store overlap block `α_k` for current iteration."""
function calcoverlap!(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim; 
    blocksize = fact.blocksize;
    α_k = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim);

    vec = fact.v_last
    mul!(α_k, vec', fact.r);

    α_k .= Hermitian(α_k);
end


"""Advance block-Lanczos tridiagonalization by one iteration."""
function tridiagonalization!(fact::BlockLanczosFactorization)
    
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    β_  = view(fact.mat, krylovdim+1:krylovdim+blocksize, krylovdim-blocksize+1:krylovdim);
    β′_ = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim+1:krylovdim+blocksize);

    QR = qr(fact.r)
    β = QR.R


    copyto!(β_, β)    
    copyto!(β′_, β')    

    update!(fact, fact.pu.Matrix(QR.Q))
end
