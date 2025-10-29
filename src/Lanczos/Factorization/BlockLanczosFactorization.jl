

mutable struct BlockLanczosFactorization{E,P} <: KrylovFactorization{E,P}
    krylovdim::Int # current Krylov dimension
    blocksize::Int # block dimension
    basis::OrthonormalBasis # basis of length k
    r::  AbstractMatrix{E}
    v_last:: AbstractMatrix{E}
    v_secondlast:: AbstractMatrix{E}
    mat::AbstractMatrix{E}
    pu::P
    
    function BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, v0::AbstractMatrix{E}, pu::P) where {E, P<:ProcessingUnit}
        r               = similar(v0)
        v_last          = pu.zeros(E, size(v0))
        v_secondlast    = pu.zeros(E, size(v0))
        mat = pu.zeros(E, maxdim, maxdim)

        return new{E,P}(0, size(v0, 2), basis, r, v_last, v_secondlast, mat, pu)
    end
    
end



# get factorization properties

function getoverlap(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    α = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim);
    return α
    
end

function getnorm(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    β = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-2blocksize+1:krylovdim-blocksize);
    return β
    
end


# add factorization properties 

function addnorm!(fact::BlockLanczosFactorization, β::AbstractMatrix)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim

    β_  = view(fact.mat, krylovdim-blocksize+1:krylovdim   , krylovdim-2blocksize+1:krylovdim-blocksize);
    β′_ = view(fact.mat, krylovdim-2blocksize+1:krylovdim-blocksize, krylovdim-blocksize+1:krylovdim   );

    copyto!(β_, β)
    copyto!(β′_, β')    
end




# other functions

function constructfactorizedmat(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim
    factmat = view(fact.mat, 1:krylovdim, 1:krylovdim)
    # display(factmat)

    return isa(fact.pu, GPU) ? CuMatrix(factmat) : factmat
end

function calc_norm_krylovvec!(fact::BlockLanczosFactorization)
    # QR = qr!(factorization.r)
    QR = qr(fact.r)

    β = QR.R
    copy!(fact.r, fact.pu.Matrix(QR.Q))

    return β
end


function calcoverlap!(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim; 
    blocksize = fact.blocksize;
    α_k = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim);

    vec = fact.v_last
    mul!(α_k, vec', fact.r);

    α_k .= Symmetric(α_k);
end


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
