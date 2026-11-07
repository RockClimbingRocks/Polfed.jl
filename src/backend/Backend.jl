# Backend/ — infrastructure: GPU dispatch plumbing + the CPU/GPU processing-unit
# types. Included at the `Polfed` top level (NOT a submodule), so the CUDA
# extension's `Polfed._set_cuda_backend!` / `Polfed.is_gpu_array` hooks resolve
# unchanged. `ProcessingUnit` depends on the GPU accessors, so order matters.

include("gpu_dispatch.jl")
include("ProcessingUnit.jl")
include("library_threads.jl")   # with_serial_libraries (BLAS + MKL thread scoping)
include("openmp.jl")            # with_openmp_blocktime (MKL idle-thread spinning)
