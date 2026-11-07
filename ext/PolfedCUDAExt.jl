module PolfedCUDAExt

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
import Polfed

_cuda_memory_info() = CUDA.memory_info()
_cuda_available_memory() = first(CUDA.memory_info())
_cuda_total_memory() = last(CUDA.memory_info())
_cuda_device_count() = isdefined(CUDA, :devices) ? length(CUDA.devices()) : 0

function __init__()
    Polfed._set_cuda_backend!(
        array=CUDA.CuArray,
        matrix=CUDA.CuMatrix,
        vector=CUDA.CuVector,
        matrix_type=CUDA.CuMatrix,
        vector_type=CUDA.CuVector,
        zeros=CUDA.zeros,
        ones=CUDA.ones,
        rand=CUDA.rand,
        randn=CUDA.randn,
        memory_info=_cuda_memory_info,
        available_memory=_cuda_available_memory,
        total_memory=_cuda_total_memory,
        functional=CUDA.functional,
        device_count=_cuda_device_count,
        synchronize=CUDA.synchronize,
        reclaim=CUDA.reclaim,
        allowscalar=(f -> CUDA.@allowscalar f()),
    )
    return nothing
end

# Views, adjoints and reshapes of device arrays count too: Lanczos hands those to the helpers that
# decide where a result lives.
Polfed.is_gpu_array(::CUDA.AnyCuArray) = true
# The concrete types, not `AbstractCuSparseMatrix`: CUDA.jl 6 no longer makes them subtypes of it.
Polfed.is_gpu_sparse_matrix(::Union{CUSPARSE.CuSparseMatrixCSR,CUSPARSE.CuSparseMatrixCSC,
                                    CUSPARSE.CuSparseMatrixCOO,CUSPARSE.CuSparseMatrixBSR}) = true

include("cusparse_operator.jl")

end
