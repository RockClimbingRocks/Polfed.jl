
"""
    HybridMatrixBasis(maxdim_gpu::Int, maxdim_cpu::Int, x0::AbstractVecOrMat{T}) where {T}

Two-tier basis storage backend: fills GPU matrix columns first, then spills to
CPU matrix columns.
"""
mutable struct HybridMatrixBasis{T<:Number, MG<:AbstractMatrix{T}, MC<:AbstractMatrix{T}} <: OrthonormalBasis
    gpu_basis::MG         # GPU storage
    cpu_basis::MC         # CPU storage
    nvecs_gpu::Integer        # number of vectors currently on GPU
    nvecs_cpu::Integer        # number of vectors currently on CPU
    blocksize::Integer


    """Build a hybrid GPU/CPU basis with fixed GPU and CPU column capacities."""
    function HybridMatrixBasis(maxdim_gpu::Int, maxdim_cpu::Int, x0::AbstractVecOrMat{T}) where {T}
        cuda_available() || error("CUDA is not available; HybridMatrixBasis requires a GPU.")
        hilbertspacedim = size(x0, 1)
        blocksize        = size(x0, 2)

        gpu_basis = gpu_zeros(T, hilbertspacedim, maxdim_gpu)
        cpu_basis = zeros(T, hilbertspacedim, maxdim_cpu)
        MG = typeof(gpu_basis)
        MC = typeof(cpu_basis)

        new{T, MG, MC}(gpu_basis, cpu_basis, 0, 0, blocksize)
    end
end


"""
    add!(basis::HybridMatrixBasis, v::AbstractVecOrMat) -> nothing

Append vectors/blocks to hybrid storage (GPU first, then CPU).
"""
function add!(basis::HybridMatrixBasis, v::AbstractVecOrMat{T}) where {T}
    # Convert vector to 2D form if needed
    mat_v = ndims(v) == 1 ? reshape(v, :, 1) : v
    ncols = size(mat_v, 2)

    # Try GPU first
    if basis.nvecs_gpu + ncols <= size(basis.gpu_basis, 2)
        basis.gpu_basis[:, basis.nvecs_gpu+1 : basis.nvecs_gpu+ncols] .= mat_v
        basis.nvecs_gpu += ncols

    # Then CPU
    elseif basis.nvecs_cpu + ncols <= size(basis.cpu_basis, 2)
        if basis.nvecs_cpu == 0
            polfed_log(POLFED_INFO_LEVEL, "HybridMatrixBasis switching to CPU basis storage.")
        end
        basis.cpu_basis[:, basis.nvecs_cpu+1 : basis.nvecs_cpu+ncols] .= Array(mat_v) # move to CPU
        basis.nvecs_cpu += ncols

    # Out of space
    else
        error("HybridMatrixBasis is full (GPU + CPU)")
    end
end


"""
    all(B::HybridMatrixBasis)

Return stored basis:
- GPU view only if no CPU spill happened,
- tuple `(gpu_view, cpu_view)` when both tiers are used.
"""
function all(B::HybridMatrixBasis)
    totalvecs = B.nvecs_gpu + B.nvecs_cpu
    if totalvecs == 0
        throw(ArgumentError("Basis is empty"))
    end

    if B.nvecs_cpu == 0
        return view(B.gpu_basis, :, 1:B.nvecs_gpu)
    else
        # Return tuple (GPU part, CPU part)
        return (
            view(B.gpu_basis, :, 1:B.nvecs_gpu),
            view(B.cpu_basis, :, 1:B.nvecs_cpu)
        )
    end
end

"""
    all_withoutlasttwo(B::HybridMatrixBasis)

Return basis storage excluding the newest two blocks, preserving hybrid tuple
layout when both GPU and CPU portions are needed.
"""
function all_withoutlasttwo(B::HybridMatrixBasis)
    blocksize = B.blocksize
    totalvecs = B.nvecs_gpu + B.nvecs_cpu

    if totalvecs < 2 * blocksize
        throw(ArgumentError("Not enough vectors in the basis"))
    end

    needed = totalvecs - 2blocksize

    if B.nvecs_cpu == 0
        # All still on GPU
        return view(B.gpu_basis, :, 1:needed)
    elseif needed <= B.nvecs_gpu
        # Only GPU part is needed
        return view(B.gpu_basis, :, 1:needed)
    else
        # Both GPU and CPU parts are needed
        gpu_part = view(B.gpu_basis, :, 1:B.nvecs_gpu)
        cpu_part = view(B.cpu_basis, :, 1:needed - B.nvecs_gpu)
        return (gpu_part, cpu_part)
    end
end
