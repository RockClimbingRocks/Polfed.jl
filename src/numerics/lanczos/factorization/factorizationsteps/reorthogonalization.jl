"""
    reorthogonalization!(factorization::KrylovFactorization, strategy::ReOrthTechnique) -> nothing

Apply reorthogonalization to the Lanczos residual workspace according to the
selected strategy:
- [`FullRO`](@ref): every iteration (except trivial early basis sizes),
- [`PartialRO`](@ref): periodically, according to `skip`.
"""
@inline function reorthogonalization!(factorization::KrylovFactorization, ::FullRO)
    krylovbasis = all_withoutlasttwo(factorization.basis)
    isempty(krylovbasis) && return   # MatrixBasis: 0-vector view; VectorBasis: 0 entries
    _reorthogonalization!(factorization, krylovbasis)
end

"""`PartialRO` scheduling overload of `reorthogonalization!`."""
@inline function reorthogonalization!(factorization::KrylovFactorization, pro::PartialRO)
    krylovdim = factorization.krylovdim
    blocksize = factorization isa LanczosFactorization ? 1 : factorization.blocksize
    kryloviter = krylovdim ÷ blocksize
    pro_condition = (kryloviter % (2+pro.skip) == 0) || ((kryloviter-1) % (2+pro.skip) == 0)

    if pro_condition && krylovdim > 2*blocksize
        krylovbasis = all_withoutlasttwo(factorization.basis)
        _reorthogonalization!(factorization, krylovbasis)
    end
end

# A MatrixBasis is one dense matrix, so its projection can use a single reusable BLAS
# workspace.  Only the prefix corresponding to the currently accumulated basis is active.
@inline function _reorthogonalization!(factorization::KrylovFactorization,
                                       krylovbasis::AbstractMatrix)
    coeff = _active_reorth_coeff(factorization.layout, factorization.reorth_coeff,
                                 krylovbasis)
    reorthogonalization!(factorization.layout, krylovbasis, factorization.r, coeff,
                         factorization.pu)
end

# A VectorBasis is a vector of separately allocated vectors/blocks and retains its existing
# one-basis-entry-at-a-time implementation.  It deliberately has no maxdim-sized workspace.
@inline function _reorthogonalization!(factorization::KrylovFactorization,
                                       krylovbasis::AbstractVector{<:AbstractVecOrMat})
    reorthogonalization!(factorization.layout, krylovbasis, factorization.r,
                         factorization.pu)
end

@inline _active_reorth_coeff(::ColumnLayout, coeff::AbstractVector, Q::AbstractMatrix) =
    view(coeff, 1:size(Q, 2))
@inline _active_reorth_coeff(::ColumnLayout, coeff::AbstractMatrix, Q::AbstractMatrix) =
    view(coeff, 1:size(Q, 2), :)
@inline _active_reorth_coeff(::RowLayout, coeff::AbstractMatrix, Q::AbstractMatrix) =
    view(coeff, :, 1:size(Q, 1))

# --- Matrix Gram-Schmidt reorthogonalization kernel (MatrixBasis) ---
"""
    reorthogonalization!(layout, krylovbasis::AbstractMatrix, W, pu) -> nothing

Low-level Gram-Schmidt projection: subtract the component of `W` in the span of
`krylovbasis`, with the multiplication side chosen by `layout` ([`reorthogonalize!`](@ref)).
`W` is mutated in place.
"""
@inline function reorthogonalization!(layout::Layout, krylovbasis::AbstractMatrix{T}, W::AbstractVecOrMat{T}, ::ProcessingUnit) where {T<:Number}
    reorthogonalize!(layout, W, krylovbasis)
    return nothing
end

"""
    reorthogonalization!(layout, krylovbasis, W, coeff, pu) -> nothing

Allocation-free `MatrixBasis` kernel using the caller-owned active coefficient workspace
`coeff`. This is the path used by `LanczosFactorization` and `BlockLanczosFactorization`.
"""
@inline function reorthogonalization!(layout::Layout, krylovbasis::AbstractMatrix{T},
                                      W::AbstractVecOrMat{T}, coeff::AbstractVecOrMat{T},
                                      ::ProcessingUnit) where {T<:Number}
    reorthogonalize!(layout, W, krylovbasis, coeff)
    return nothing
end

# --- Vector-of-arrays Gram-Schmidt kernel (VectorBasis) ---
"""
    reorthogonalization!(layout, krylovbasis::AbstractVector{<:AbstractVecOrMat}, W, pu) -> nothing

[`VectorBasis`](@ref) overload: project `W` against each stored basis vector/block in
turn (column semantics; `VectorBasis` is used only with [`ColumnLayout`](@ref)). `W` is
mutated in place.
"""
@inline function reorthogonalization!(::Layout,
                                      krylovbasis::AbstractVector{<:AbstractVector},
                                      W::AbstractVector, ::ProcessingUnit)
    for v in krylovbasis
        axpy!(-dot(v, W), v, W)
    end
    return nothing
end

@inline function reorthogonalization!(::Layout,
                                      krylovbasis::AbstractVector{<:AbstractMatrix{T}},
                                      W::AbstractMatrix{T}, pu::ProcessingUnit) where {T<:Number}
    isempty(krylovbasis) && return nothing
    bs = size(W, 2)
    coeff = pu.Matrix{T}(undef, bs, bs)
    for v in krylovbasis
        mul!(coeff, v', W)
        mul!(W, v, coeff, -one(T), one(T))
    end
    return nothing
end
