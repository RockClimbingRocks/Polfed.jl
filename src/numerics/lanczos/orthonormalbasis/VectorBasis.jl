"""
    VectorBasis{T,E}(maxdim::Int, x0::AbstractVecOrMat, pu::ProcessingUnit)

Vector-of-arrays basis storage backend for Lanczos vectors/blocks.
"""
mutable struct VectorBasis{T,E} <: OrthonormalBasis
    basis::Vector{T}
    nvecs::Int
    blocksize::Int

    function VectorBasis{E}(maxdim::Int, x0::T, ::ProcessingUnit,
                            layout::Layout=ColumnLayout()) where {T<:AbstractVecOrMat, E<:Number}
        bs = blocksize(layout, x0)
        # `maxdim` is a per-vector budget (matching MatrixBasis); each stored entry is a
        # full block of `bs` vectors, so allocate `maxdim / bs` entries.
        nentries = cld(maxdim, bs)
        basis = [similar(x0) for _ in 1:nentries]
        new{T,E}(basis, 0, bs)
    end
end

"""
    add!(basis::VectorBasis, v) -> nothing

Append a vector/block to basis storage, shifting out the oldest entry when full.
"""
function add!(basis::VectorBasis{T,E}, v::T) where {T,E}
    if basis.nvecs < length(basis.basis)
        basis.nvecs += 1
        basis.basis[basis.nvecs] .= v
    else
        for i in 1:basis.nvecs-1
            basis.basis[i] .= basis.basis[i+1]
        end
        basis.basis[basis.nvecs] .= v
    end

    return nothing
end

"""
    allvecs(basis::VectorBasis)

Return currently stored basis entries.
"""
function allvecs(basis::VectorBasis)
    basis.nvecs == 0 && throw(ArgumentError("Basis is empty"))
    return basis.basis[1:basis.nvecs]
end

"""
    all_withoutlasttwo(basis::VectorBasis)

Return basis entries excluding the most recent two blocks. `nvecs` counts whole
blocks (entries) here, so the two most recent blocks are simply the last two entries.
"""
function all_withoutlasttwo(basis::VectorBasis)
    basis.nvecs < 2 && throw(ArgumentError("Not enough vectors in the basis"))
    return basis.basis[1:basis.nvecs-2]
end
