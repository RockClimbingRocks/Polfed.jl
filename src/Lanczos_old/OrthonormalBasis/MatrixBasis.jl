mutable struct MatrixBasis{T,E} <: OrthonormalBasis 
    basis::AbstractMatrix{E}
    nvecs::Int
    blockdim::Int

    function MatrixBasis{E}(maxdim::Int, x0::T, pu::ProcessingUnit) where {T<:AbstractVecOrMat, E<:Real}
        hilbertspacedim = size(x0,1)
        blockdim = size(x0,2)
        basis = pu.mat{E}(undef, hilbertspacedim, maxdim)
        nvecs = 0
        new{T,E}(basis, nvecs, blockdim)
    end
end

function add!(basis::MatrixBasis, v::AbstractVector)
    if basis.nvecs < size(basis.basis, 2)
        basis.nvecs += 1
        basis.basis[:, basis.nvecs] .= v
    else
        # Shift all columns to the left
        for i in 1:basis.nvecs-1
            basis.basis[:, i] .= basis.basis[:, i+1]
        end
        # Add the new vector to the last column
        basis.basis[:, basis.nvecs] .= v
    end
end

function add!(basis::MatrixBasis, v::AbstractMatrix)
    ncols = size(v, 2)
    if basis.nvecs + ncols <= size(basis.basis, 2)
        basis.basis[:, basis.nvecs+1:basis.nvecs+ncols] .= v
        basis.nvecs += ncols
    else
        shift = basis.nvecs + ncols - size(basis.basis, 2)
        # Shift columns to the left
        for i in 1:basis.nvecs-shift
            basis.basis[:, i] .= basis.basis[:, i+shift]
        end
        # Add the new matrix to the last columns
        basis.basis[:, basis.nvecs-shift+1:basis.nvecs-shift+ncols] .= v
        basis.nvecs = size(basis.basis, 2)
    end
end

function last(basis::MatrixBasis)
    blockdim = basis.blockdim
    if blockdim > basis.nvecs || blockdim <= 0
        throw(ArgumentError("Invalid number of vectors requested"))
    end
    return blockdim == 1 ? view(basis.basis, :, basis.nvecs) : view(basis.basis, :, basis.nvecs-blockdim+1:basis.nvecs)
end

function secondlast(basis::MatrixBasis)
    blockdim = basis.blockdim
    if basis.nvecs < 2 * blockdim
        throw(ArgumentError("Not enough vectors in the basis"))
    end
    return blockdim == 1 ? view(basis.basis, :, basis.nvecs-1) : view(basis.basis, :, basis.nvecs-2blockdim+1:basis.nvecs-blockdim)
end

function all(basis::MatrixBasis)
    if basis.nvecs == 0
        throw(ArgumentError("Basis is empty"))
    end
    return view(basis.basis, :, 1:basis.nvecs)
end

function all_withoutlasttwo(basis::MatrixBasis)
    blockdim = basis.blockdim
    if basis.nvecs < 2*blockdim
        throw(ArgumentError("Not enough vectors in the basis"))
    end
    return view(basis.basis, :, 1:basis.nvecs-2blockdim)
end