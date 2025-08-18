abstract type OrthonormalBasis end

include("MatrixBasis.jl")
include("HybridMatrixBasis.jl")
include("VectorBasis.jl")


function createbasis(maxdim::Int, x0::AbstractVecOrMat{E},
                     basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit) where {E<:Real}

    if basistype === HybridMatrixBasis
        # Bytes needed for one vector
        bytes_per_vector = sizeof(E) * size(x0, 1)
        
        # Get GPU free memory
        # free_mem, _ = CUDA.memory_status()
        CUDA.memory_status()  

        # Programmatically get free and total memory
        free_mem = CUDA.available_memory()    # bytes available for allocation
        total_mem = CUDA.total_memory()       # total GPU memory
        used_mem = total_mem - free_mem
        safety_factor = 0.9  # leave headroom
        usable_mem = free_mem * safety_factor
        # usable_mem = 2. * 1e9
        # Max vectors we can fit on GPU
        max_vectors_gpu = floor(Int, usable_mem / bytes_per_vector)
        
        # Respect total maxdim
        maxdim_gpu = min(max_vectors_gpu, maxdim)
        maxdim_cpu = maxdim - maxdim_gpu
        
        if maxdim_gpu == 0
            @warn "No GPU memory available for basis vectors — storing all on CPU"
        end

        
        println("Reserved memmory of OB type HybridMatrixBasis: ")
        println("   Free GPU memmory: $(free_mem / 1e9) GB")
        println("   Reserved GPU memmory: $(maxdim_gpu * bytes_per_vector / 1e9) GB")
        println("   Number of vectors reserved: $(maxdim_gpu)")
        println("   Free CPU memmory: $(Sys.free_memory() / 1e9) GB")
        println("   Number of vectors reserved: $(maxdim_cpu)")
        println("   Reserved CPU memmory: $(maxdim_cpu * bytes_per_vector / 1e9) GB")


        
        return HybridMatrixBasis(maxdim_gpu, maxdim_cpu, x0)
    else
        return basistype{E}(maxdim, x0, pu)
    end
end
