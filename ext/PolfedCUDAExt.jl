module PolfedCUDAExt

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

import Polfed
import Polfed.PolfedCore: materialize_host_matrix
import Polfed.PolfedCore: optimized_clenshaw_final_sum!,
    optimized_clenshaw_recurrence_relation!, optimized_mapping!
using Polfed.PolfedCore: Parallelization

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

Polfed.is_gpu_array(::CUDA.CuArray) = true
Polfed.is_gpu_sparse_matrix(::CUDA.CUSPARSE.AbstractCuSparseMatrix) = true

function materialize_host_matrix(mat::CUDA.CUSPARSE.AbstractCuSparseMatrix{T}) where {T<:Number}
    I_gpu, J_gpu, V_gpu = findnz(mat)
    I = Int.(collect(I_gpu))
    J = Int.(collect(J_gpu))
    V = collect(V_gpu)
    return sparse(I, J, V, size(mat, 1), size(mat, 2))
end

include(joinpath(dirname(pathof(Polfed)), "Optimization", "gpu", "vector_dispatch.jl"))

end
