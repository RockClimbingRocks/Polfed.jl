abstract type OrthonormalBasis end

include("MatrixBasis.jl")
include("HybridMatrixBasis.jl")
include("VectorBasis.jl")


"""
    createbasis(maxdim::Int, x0::AbstractVecOrMat{E}, basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit) where {E<:Number} -> OrthonormalBasis

Create the concrete basis storage object used by Lanczos iterations.

For `HybridMatrixBasis`, this function also estimates GPU-resident capacity
from currently available GPU memory.
"""
function createbasis(maxdim::Int, x0::AbstractVecOrMat{E},
                     basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit) where {E<:Number}

    if basistype === HybridMatrixBasis
        cuda_available() || error("CUDA is not available; HybridMatrixBasis requires a GPU.")
        # Bytes needed for one vector
        bytes_per_vector = sizeof(E) * size(x0, 1)
        
        # Get GPU free memory
        # Programmatically get free and total memory
        free_mem = gpu_available_memory()    # bytes available for allocation
        total_mem = gpu_total_memory()       # total GPU memory
        safety_factor = 0.9  # leave headroom
        usable_mem = free_mem * safety_factor
        # usable_mem = 2. * 1e9
        # Max vectors we can fit on GPU
        max_vectors_gpu = floor(Int, usable_mem / bytes_per_vector)
        
        # Respect total maxdim
        maxdim_gpu = min(max_vectors_gpu, maxdim)
        maxdim_cpu = maxdim - maxdim_gpu
        
        if maxdim_gpu == 0
            polfed_log(POLFED_WARN_LEVEL, "No GPU memory available for basis vectors; storing all on CPU.")
        end

        
        free_gpu_gb = free_mem / 1e9
        reserved_gpu_gb = maxdim_gpu * bytes_per_vector / 1e9
        free_cpu_gb = Sys.free_memory() / 1e9
        reserved_cpu_gb = maxdim_cpu * bytes_per_vector / 1e9

        polfed_log(
            POLFED_INFO_LEVEL,
            "HybridMatrixBasis reservation.",
            free_gpu_gb=free_gpu_gb,
            reserved_gpu_gb=reserved_gpu_gb,
            vectors_gpu=maxdim_gpu,
            free_cpu_gb=free_cpu_gb,
            reserved_cpu_gb=reserved_cpu_gb,
            vectors_cpu=maxdim_cpu,
        )

        return HybridMatrixBasis(maxdim_gpu, maxdim_cpu, x0)
    else
        return basistype{E}(maxdim, x0, pu)
    end
end
