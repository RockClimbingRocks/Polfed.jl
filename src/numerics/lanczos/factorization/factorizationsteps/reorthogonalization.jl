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
    reorthogonalization!(factorization.layout, krylovbasis, factorization.r, factorization.pu)
end

"""`PartialRO` scheduling overload of `reorthogonalization!`."""
@inline function reorthogonalization!(factorization::KrylovFactorization, pro::PartialRO)
    krylovdim = factorization.krylovdim
    blocksize = factorization isa LanczosFactorization ? 1 : factorization.blocksize
    kryloviter = krylovdim ÷ blocksize
    pro_condition = (kryloviter % (2+pro.skip) == 0) || ((kryloviter-1) % (2+pro.skip) == 0)

    if pro_condition && krylovdim > 2*blocksize
        krylovbasis = all_withoutlasttwo(factorization.basis)
        reorthogonalization!(factorization.layout, krylovbasis, factorization.r, factorization.pu)
    end
end

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

# --- Vector-of-arrays Gram-Schmidt kernel (VectorBasis) ---
"""
    reorthogonalization!(layout, krylovbasis::AbstractVector{<:AbstractVecOrMat}, W, pu) -> nothing

[`VectorBasis`](@ref) overload: project `W` against each stored basis vector/block in
turn (column semantics; `VectorBasis` is used only with [`ColumnLayout`](@ref)). `W` is
mutated in place.
"""
@inline function reorthogonalization!(::Layout, krylovbasis::AbstractVector{<:AbstractVecOrMat}, W::AbstractVecOrMat, ::ProcessingUnit)
    for v in krylovbasis
        W .-= v * (v' * W)
    end
end
