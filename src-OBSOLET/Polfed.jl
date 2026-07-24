module Polfed

using Distributed

using LinearAlgebra, UnPack, Base.Threads, QuadGK, Printf, SharedArrays, SparseArrays, Logging
# using KrylovKit
# @everywhere using LinearAlgebra
using StaticArrays: SVector

CUDA_AVAILABLE = false

cuda_available() = CUDA_AVAILABLE
_set_cuda_available!(available::Bool) = (@eval CUDA_AVAILABLE = $available; nothing)
_cuda_unavailable() = error("CUDA support is not active. Install and load CUDA.jl (`using CUDA`) before passing CUDA arrays to Polfed.")

const _GPU_ARRAY = Ref{Any}(nothing)
const _GPU_MATRIX = Ref{Any}(nothing)
const _GPU_VECTOR = Ref{Any}(nothing)
const _GPU_MATRIX_TYPE = Ref{Any}(nothing)
const _GPU_VECTOR_TYPE = Ref{Any}(nothing)
const _GPU_ZEROS = Ref{Any}(nothing)
const _GPU_ONES = Ref{Any}(nothing)
const _GPU_RAND = Ref{Any}(nothing)
const _GPU_RANDN = Ref{Any}(nothing)
const _GPU_MEMORY_INFO = Ref{Any}(nothing)
const _GPU_AVAILABLE_MEMORY = Ref{Any}(nothing)
const _GPU_TOTAL_MEMORY = Ref{Any}(nothing)
const _GPU_FUNCTIONAL = Ref{Any}(nothing)
const _GPU_DEVICE_COUNT = Ref{Any}(nothing)
const _GPU_SYNCHRONIZE = Ref{Any}(nothing)
const _GPU_RECLAIM = Ref{Any}(nothing)
const _GPU_ALLOWSCALAR = Ref{Any}(nothing)

function _set_cuda_backend!(;
    array,
    matrix,
    vector,
    matrix_type,
    vector_type,
    zeros,
    ones,
    rand,
    randn,
    memory_info,
    available_memory,
    total_memory,
    functional,
    device_count,
    synchronize,
    reclaim,
    allowscalar,
)
    _GPU_ARRAY[] = array
    _GPU_MATRIX[] = matrix
    _GPU_VECTOR[] = vector
    _GPU_MATRIX_TYPE[] = matrix_type
    _GPU_VECTOR_TYPE[] = vector_type
    _GPU_ZEROS[] = zeros
    _GPU_ONES[] = ones
    _GPU_RAND[] = rand
    _GPU_RANDN[] = randn
    _GPU_MEMORY_INFO[] = memory_info
    _GPU_AVAILABLE_MEMORY[] = available_memory
    _GPU_TOTAL_MEMORY[] = total_memory
    _GPU_FUNCTIONAL[] = functional
    _GPU_DEVICE_COUNT[] = device_count
    _GPU_SYNCHRONIZE[] = synchronize
    _GPU_RECLAIM[] = reclaim
    _GPU_ALLOWSCALAR[] = allowscalar
    _set_cuda_available!(true)
    return nothing
end

_gpu_backend(ref::Base.RefValue{Any}) = ref[] === nothing ? _cuda_unavailable() : ref[]
_gpu_call(ref::Base.RefValue{Any}, args...; kwargs...) = _gpu_backend(ref)(args...; kwargs...)

"""
    is_gpu_array(x) -> Bool

Return `true` when CUDA is available and `x` is a CUDA array.
"""
is_gpu_array(x) = false
is_gpu_sparse_matrix(x) = false

gpu_array(args...; kwargs...) = _gpu_call(_GPU_ARRAY, args...; kwargs...)
gpu_matrix(args...; kwargs...) = _gpu_call(_GPU_MATRIX, args...; kwargs...)
gpu_vector(args...; kwargs...) = _gpu_call(_GPU_VECTOR, args...; kwargs...)
gpu_matrix_type() = _gpu_backend(_GPU_MATRIX_TYPE)
gpu_vector_type() = _gpu_backend(_GPU_VECTOR_TYPE)
gpu_zeros(args...; kwargs...) = _gpu_call(_GPU_ZEROS, args...; kwargs...)
gpu_ones(args...; kwargs...) = _gpu_call(_GPU_ONES, args...; kwargs...)
gpu_rand(args...; kwargs...) = _gpu_call(_GPU_RAND, args...; kwargs...)
gpu_randn(args...; kwargs...) = _gpu_call(_GPU_RANDN, args...; kwargs...)
gpu_memory_info() = _gpu_call(_GPU_MEMORY_INFO)
gpu_available_memory() = _gpu_call(_GPU_AVAILABLE_MEMORY)
gpu_total_memory() = _gpu_call(_GPU_TOTAL_MEMORY)
gpu_functional() = _GPU_FUNCTIONAL[] === nothing ? false : _GPU_FUNCTIONAL[]()
gpu_device_count() = _GPU_DEVICE_COUNT[] === nothing ? 0 : _GPU_DEVICE_COUNT[]()
gpu_synchronize() = _GPU_SYNCHRONIZE[] === nothing ? nothing : _GPU_SYNCHRONIZE[]()
gpu_reclaim() = _GPU_RECLAIM[] === nothing ? nothing : _GPU_RECLAIM[]()
gpu_allowscalar(f::F) where {F<:Function} = _GPU_ALLOWSCALAR[] === nothing ? f() : _GPU_ALLOWSCALAR[](f)

include("Common/common.jl")
include("QSun/QSun.jl")
include("Models/Models.jl")
include("Clenshaw/ClenshawMapping.jl")
include("Lanczos/Lanczos.jl")

import .Common: Formatter, fmt, bold, cyan, blue, green, red, yellow, @addtime!
import .ClenshawMapping: Clenshaw
import .Lanczos: lanczos, lanczos_extrema, FactorizationReport, display_factorization_report,
        FullRO, PartialRO, ReOrthTechnique,
        MatrixBasis, HybridMatrixBasis, OrthonormalBasis

const main_module_file = abspath(@__FILE__)

include("Polfed/PolfedCore.jl")
using .PolfedCore: polfed, display_report, MappingConfig, TransformConfig, DoSConfig, FactorizationConfig,
    MulColsParallel, TwoLevelParallel, NoParallel, PolfedDefaults, CPU, GPU,
    optimized_mapping!, optimized_clenshaw_recurrence_relation!, optimized_clenshaw_final_sum!
export polfed, display_report
export lanczos_extrema
export MappingConfig, TransformConfig, DoSConfig, FactorizationConfig
export MulColsParallel, TwoLevelParallel, NoParallel
export CPU, GPU
export optimized_mapping!, optimized_clenshaw_recurrence_relation!, optimized_clenshaw_final_sum!

end # module
