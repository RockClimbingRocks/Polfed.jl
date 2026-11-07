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
    mat::Matrix{T}                 # projected block-tridiagonal matrix — always on the host
    orth_coeff::AbstractMatrix{T}  # bs × bs on the vectors' device: overlaps, DGKS δ, α for project!
    coeff_scratch::AbstractMatrix{T}  # bs × bs on the vectors' device: β for project!
    reorth_coeff::Union{Nothing,AbstractMatrix{T}}
    pu::P
    layout::L                      # block storage orientation (Column/Row)

    """Build initial block-Lanczos factorization state."""
    function BlockLanczosFactorization(maxdim::Int, basis::OrthonormalBasis, v0::AbstractMatrix{T}, pu::P,
                                       layout::L=ColumnLayout()) where {T<:Number, P<:ProcessingUnit, L<:Layout}
        S = real(T)
        r = similar(v0)
        v_last = pu.zeros(T, size(v0))
        v_secondlast = pu.zeros(T, size(v0))
        mat = zeros(T, maxdim, maxdim)
        bs = blocksize(layout, v0)
        orth_coeff = pu.zeros(T, bs, bs)
        coeff_scratch = pu.zeros(T, bs, bs)
        reorth_coeff = if basis isa MatrixBasis
            layout isa RowLayout ? pu.zeros(T, bs, maxdim) : pu.zeros(T, maxdim, bs)
        else
            nothing
        end

        return new{S,T,P,L}(0, bs, basis, r, v_last, v_secondlast, mat, orth_coeff, coeff_scratch,
                            reorth_coeff, pu, layout)

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

"""Accumulate a DGKS correction into the stored overlap block `α_k`."""
@inline function addoverlap!(fact::BlockLanczosFactorization, δα::AbstractMatrix)
    α = getoverlap(fact)
    α .+= δα
    α .= Hermitian(α)          # same symmetrization calcoverlap! applies
    return nothing
end


"""Projected factorized matrix view, `Hermitian` on the host (also for a GPU run)."""
function constructfactorizedmat(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim
    return Hermitian(view(fact.mat, 1:krylovdim, 1:krylovdim))
end

"""
    _device_coeff(fact, M, slot) -> AbstractMatrix

A host coupling block `M` (a view of `fact.mat`, or its adjoint) in a form [`project!`](@ref)
can apply to the factorization's vectors: `M` itself on the CPU, a copy in device scratch on
the GPU (`slot` 1 → `orth_coeff`, 2 → `coeff_scratch`, so two blocks can be live at once).
"""
function _device_coeff(fact::BlockLanczosFactorization, M::AbstractMatrix, slot::Int)
    fact.pu isa GPU || return M
    return copyto!(slot == 1 ? fact.orth_coeff : fact.coeff_scratch, Matrix(M))
end

"""Orthonormalize the residual block per `layout` (column: `qr`→`R`; row: `lq`→`L`) and return
the `bs × bs` coupling factor."""
function calc_norm_krylovvec!(fact::BlockLanczosFactorization)
    Q, β = orthonormalize(fact.layout, fact.r)
    copy!(fact.r, Q)
    return β
end

"""Compute and store the overlap block `α_k` for the current iteration (`overlap` per layout)."""
function calcoverlap!(fact::BlockLanczosFactorization)
    krylovdim = fact.krylovdim
    blocksize = fact.blocksize
    α_k = view(fact.mat, krylovdim-blocksize+1:krylovdim, krylovdim-blocksize+1:krylovdim)
    if fact.pu isa GPU                     # the product lands on the device; the block lives here
        overlap!(fact.layout, fact.orth_coeff, fact.v_last, fact.r)
        copyto!(α_k, Array(fact.orth_coeff))
    else
        overlap!(fact.layout, α_k, fact.v_last, fact.r)
    end
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

    update!(fact, Q)
    return true
end
