"""
    BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, v0::AbstractMatrix{T}, pu::ProcessingUnit)

State container for block Lanczos factorization.
"""
mutable struct BlockLanczosFactorization{S<:Real,T<:Number,P,L<:Layout} <: KrylovFactorization{S,T,P}
    krylovdim::Int                 # current Krylov dimension
    blocksize::Int                 # block dimension
    basis::OrthonormalBasis        # basis of length krylovdim
    r::AbstractMatrix{T}
    v_last::AbstractMatrix{T}
    v_secondlast::AbstractMatrix{T}
    mat::AbstractMatrix{T}
    pu::P
    layout::L                      # block storage orientation (Column/Row)

    """Build initial block-Lanczos factorization state."""
    function BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, v0::AbstractMatrix{T}, pu::P,
                                       layout::L=ColumnLayout()) where {T<:Number, P<:ProcessingUnit, L<:Layout}
        S = real(T)
        r = similar(v0)
        v_last = pu.zeros(T, size(v0))
        v_secondlast = pu.zeros(T, size(v0))
        mat = pu.zeros(T, maxdim, maxdim)

        return new{S,T,P,L}(0, blocksize(layout, v0), basis, r, v_last, v_secondlast, mat, pu, layout)
    end
end

# --- overlap / norm bookkeeping ---
"""Current overlap block `α_k` (a view into the projected block-tridiagonal matrix)."""
function getoverlap(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim
    return view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim)
end

"""
Current block norm-coupling matrix `β`. Returns a zero block when only one block is
present (`krylovdim < 2·blocksize`) — the situation during a breakdown extraction,
where the single-block subspace has no coupling and the Ritz residual estimate is `0`.
"""
function getnorm(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim
    krylovdim < 2 * blocksize && return zeros(eltype(fact.mat), blocksize, blocksize)
    return view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-2blocksize+1:krylovdim-blocksize)
end

"""Store block norm-coupling matrix `β` (and its adjoint) into the projected matrix."""
function addnorm!(fact::BlockLanczosFactorization, β::AbstractMatrix)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim
    β_  = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-2blocksize+1:krylovdim-blocksize)
    β′_ = view(fact.mat, krylovdim-2blocksize+1:krylovdim-blocksize, krylovdim-blocksize+1:krylovdim)
    copyto!(β_, β)
    copyto!(β′_, β')
end

"""Projected factorized matrix view (`Hermitian` on CPU, device matrix on GPU)."""
function constructfactorizedmat(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim
    factmat = view(fact.mat, 1:krylovdim, 1:krylovdim)
    return isa(fact.pu, GPU) ? gpu_matrix(factmat) : Hermitian(factmat)
end

"""Orthonormalize the residual block per `layout` (column: `qr`→`R`; row: `lq`→`L`) and return
the `bs × bs` coupling factor."""
function calc_norm_krylovvec!(fact::BlockLanczosFactorization)
    Q, β = orthonormalize(fact.layout, fact.r)
    copy!(fact.r, fact.pu.Matrix(Q))
    return β
end

"""Compute and store the overlap block `α_k` for the current iteration (`overlap` per layout)."""
function calcoverlap!(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim
    blocksize = fact.blocksize
    α_k = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim)
    α_k .= overlap(fact.layout, fact.v_last, fact.r)
    α_k .= Hermitian(α_k)
end

"""
Advance block-Lanczos tridiagonalization by one iteration. Returns `false` on
**breakdown** (the residual block has collapsed — `‖R‖` below the breakdown
tolerance), otherwise `true`.
"""
function tridiagonalization!(fact::BlockLanczosFactorization)
    blocksize = fact.blocksize
    krylovdim = fact.krylovdim
    β_  = view(fact.mat, krylovdim+1:krylovdim+blocksize, krylovdim-blocksize+1:krylovdim)
    β′_ = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim+1:krylovdim+blocksize)

    Q, β = orthonormalize(fact.layout, fact.r)
    norm(β) < _breakdown_tol(eltype(fact.r)) && return false
    # Off-diagonal coupling of the projected (Hermitian) matrix. Column `qr` gives an
    # upper-triangular `β=R` with T[j,j+1]=β'; row `lq` gives a lower-triangular `β=L` with
    # T[j,j+1]=β — so the below/above-diagonal blocks are swapped between layouts.
    if fact.layout isa RowLayout
        copyto!(β_, β')
        copyto!(β′_, β)
    else
        copyto!(β_, β)
        copyto!(β′_, β')
    end

    update!(fact, fact.pu.Matrix(Q))
    return true
end
