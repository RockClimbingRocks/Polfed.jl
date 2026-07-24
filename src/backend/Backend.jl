# Backend/ — infrastructure: GPU dispatch plumbing + the CPU/GPU processing-unit
# types. Included at the `UniPolfed` top level (NOT a submodule), so the CUDA
# extension's `UniPolfed._set_cuda_backend!` / `UniPolfed.is_gpu_array` hooks resolve
# unchanged. `ProcessingUnit` depends on the GPU accessors, so order matters.

include("gpu_dispatch.jl")
include("ProcessingUnit.jl")
