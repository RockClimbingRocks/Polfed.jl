

mutable struct BlockLanczosFactorization{E,P} <: KrylovFactorization{E,P}
    krylovdim::Int # current Krylov dimension
    blockdim::Int # block dimension
    basis::OrthonormalBasis # basis of length k
    r::  AbstractMatrix{E}
    v_last:: AbstractMatrix{E}
    v_secondlast:: AbstractMatrix{E}
    mat::AbstractMatrix{E}
    pu::P
    
    function BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, x0::AbstractMatrix{E}, pu::P) where {E, P<:ProcessingUnit}
        r               = similar(x0)
        v_last          = pu.zeros(E, size(x0))
        v_secondlast    = pu.zeros(E, size(x0))
        mat = pu.zeros(E, maxdim, maxdim)

        return new{E,P}(0, size(x0, 2), basis, r, v_last, v_secondlast, mat, pu)
    end
    
end



# get factorization properties

function getoverlap(fact::BlockLanczosFactorization)
    blockdim = fact.blockdim
    krylovdim = fact.krylovdim

    α = view(fact.mat, krylovdim-blockdim+1:krylovdim, krylovdim-blockdim+1:krylovdim);
    return α
    
end

function getnorm(fact::BlockLanczosFactorization)
    blockdim = fact.blockdim
    krylovdim = fact.krylovdim

    β = view(fact.mat, krylovdim-blockdim+1:krylovdim, krylovdim-2blockdim+1:krylovdim-blockdim);
    return β
    
end


# add factorization properties 

function addnorm!(fact::BlockLanczosFactorization, β::AbstractMatrix)
    blockdim = fact.blockdim
    krylovdim = fact.krylovdim

    β_  = view(fact.mat, krylovdim-blockdim+1:krylovdim   , krylovdim-2blockdim+1:krylovdim-blockdim);
    β′_ = view(fact.mat, krylovdim-2blockdim+1:krylovdim-blockdim, krylovdim-blockdim+1:krylovdim   );

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
    blockdim = fact.blockdim;
    α_k = view(fact.mat, krylovdim-blockdim+1:krylovdim, krylovdim-blockdim+1:krylovdim);

    vec = fact.v_last
    mul!(α_k, vec', fact.r);

    α_k .= Symmetric(α_k);
end


function tridiagonalization!(fact::BlockLanczosFactorization)
    
    blockdim = fact.blockdim
    krylovdim = fact.krylovdim

    β_  = view(fact.mat, krylovdim+1:krylovdim+blockdim, krylovdim-blockdim+1:krylovdim);
    β′_ = view(fact.mat, krylovdim-blockdim+1:krylovdim, krylovdim+1:krylovdim+blockdim);

    QR = qr(fact.r)
    β = QR.R


    copyto!(β_, β)    
    copyto!(β′_, β')    

    update!(fact, fact.pu.Matrix(QR.Q))
end
