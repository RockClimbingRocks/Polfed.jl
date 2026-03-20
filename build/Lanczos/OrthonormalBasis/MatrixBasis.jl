"""
    MatrixBasis{E}(maxdim::Int, x0::AbstractVecOrMat, pu::ProcessingUnit)

Dense column-major basis storage for Lanczos vectors/blocks.
"""
mutable struct MatrixBasis{T,E} <: OrthonormalBasis 
    basis::AbstractMatrix{E}
    nvecs::Integer
    blocksize::Integer

    function MatrixBasis{E}(maxdim::Int, x0::T, pu::ProcessingUnit) where {T<:AbstractVecOrMat, E<:Number}
        hilbertspacedim = size(x0,1)
        blocksize = size(x0,2)
        basis = pu.Matrix{E}(undef, hilbertspacedim, maxdim)
        nvecs = 0
        new{T,E}(basis, nvecs, blocksize)
    end
end

"""
    add!(basis::MatrixBasis, v::AbstractArray) -> nothing

Append one vector or block of vectors to `basis`, shifting old columns when
capacity is exceeded.
"""
function add!(basis::MatrixBasis, v::AbstractArray)
    ncols = ndims(v) == 1 ? 1 : size(v, 2)  # number of vectors to add
    # reshape vector to a matrix for unified assignment
    vmat = ndims(v) == 1 ? reshape(v, :, 1) : v

    maxcols = size(basis.basis, 2)

    if basis.nvecs + ncols <= maxcols
        basis.basis[:, basis.nvecs+1 : basis.nvecs+ncols] .= vmat
        basis.nvecs += ncols
    else
        shift = basis.nvecs + ncols - maxcols
        # shift columns left by `shift` positions
        for i in 1:(basis.nvecs - shift)
            basis.basis[:, i] .= basis.basis[:, i + shift]
        end
        # add new columns at the end
        basis.basis[:, basis.nvecs - shift + 1 : basis.nvecs - shift + ncols] .= vmat
        basis.nvecs = maxcols
    end
end


"""
    all(basis::MatrixBasis)

Return a view of all stored basis columns.
"""
function all(basis::MatrixBasis)
    if basis.nvecs == 0
        throw(ArgumentError("Basis is empty"))
    end
    return view(basis.basis, :, 1:basis.nvecs)
end

"""
    all_withoutlasttwo(basis::MatrixBasis)

Return a view excluding the most recent two blocks (used for
reorthogonalization).
"""
function all_withoutlasttwo(basis::MatrixBasis)
    blocksize = basis.blocksize
    if basis.nvecs < 2*blocksize
        throw(ArgumentError("Not enough vectors in the basis"))
    end
    return view(basis.basis, :, 1:basis.nvecs-2blocksize)
end
