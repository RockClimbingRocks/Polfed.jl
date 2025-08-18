
mutable struct BlockLanczosFactorization{E,P} <: KrylovFactorization{E,P}
    krylovdim::Int # current Krylov dimension
    blockdim::Int # block dimension
    basis::Basis # basis of length k
    mat::AbstractMatrix{E}
    r::  AbstractMatrix{E}
    pu::P
    
    function BlockLanczosFactorization(maxdim::Int, basis::Basis, x₀::AbstractMatrix{E}, pu::P) where {E, P<:ProcessingUnit}
        r   = similar(x₀)
        mat = pu.zeros(E, maxdim, maxdim)


        # println("BLOCK Lanczos factorization test")
        # display(E)
        # display(typeof(r))
        # display(typeof(mat))
        return new{E,P}(0, size(x₀, 2), basis, mat, r, pu)
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
    copy!(fact.r, fact.pu.mat(QR.Q))

    return β
end


function calcoverlap!(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim; 
    blockdim = fact.blockdim;
    α_k = view(fact.mat, krylovdim-blockdim+1:krylovdim, krylovdim-blockdim+1:krylovdim);

    vec = last(fact.basis)
    mul!(α_k, vec', fact.r);

    α_k .= Symmetric(α_k);
end


function tridiagonalization!(fact::BlockLanczosFactorization)
    fact.krylovdim += fact.blockdim
    
    blockdim = fact.blockdim
    krylovdim = fact.krylovdim
    
    β_  = view(fact.mat, krylovdim-blockdim+1:krylovdim   , krylovdim-2blockdim+1:krylovdim-blockdim);
    β′_ = view(fact.mat, krylovdim-2blockdim+1:krylovdim-blockdim, krylovdim-blockdim+1:krylovdim   );

    QR = qr(fact.r)
    β = QR.R


    copyto!(β_, β)    
    copyto!(β′_, β')    

    add!(fact.basis, fact.pu.mat(QR.Q))
end
