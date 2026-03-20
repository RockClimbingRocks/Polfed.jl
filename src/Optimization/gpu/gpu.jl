if CUDA_AVAILABLE
    using CUDA
else
    @warn "CUDA not available; GPU optimizations are disabled."
end


include("kernels/kernels.jl")
