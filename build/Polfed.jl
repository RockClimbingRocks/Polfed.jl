module Polfed

using Distributed

using LinearAlgebra, UnPack, Base.Threads, QuadGK, Printf, SharedArrays, SparseArrays, Logging
# using KrylovKit
# @everywhere using LinearAlgebra
using StaticArrays: SVector

const CUDA_AVAILABLE = let
    try
        @eval using CUDA
        @eval using CUDA.CUSPARSE
        true
    catch err
        @warn "CUDA not available; GPU support disabled." exception=(err, catch_backtrace())
        false
    end
end

module CUDAStub
    export @allowscalar, @device_function, zeros, ones, rand, randn,
           memory_info, available_memory, total_memory, memory_status

    macro allowscalar(ex)
        return esc(ex)
    end

    macro device_function(ex)
        return esc(ex)
    end

    _unavailable() = error("CUDA is not available in this environment.")
    zeros(args...; kwargs...) = _unavailable()
    ones(args...; kwargs...) = _unavailable()
    rand(args...; kwargs...) = _unavailable()
    randn(args...; kwargs...) = _unavailable()
    memory_info() = _unavailable()
    available_memory() = _unavailable()
    total_memory() = _unavailable()
    memory_status() = _unavailable()

    module CUSPARSE
    end
end

if !CUDA_AVAILABLE
    abstract type AbstractCuArray{T,N} <: AbstractArray{T,N} end
    const CuArray = AbstractCuArray
    const CuVector = AbstractCuArray{T,1} where {T}
    const CuMatrix = AbstractCuArray{T,2} where {T}
    const CUDA = CUDAStub
end

"""
    is_gpu_array(x) -> Bool

Return `true` when CUDA is available and `x` is a CUDA array.
"""
is_gpu_array(x) = CUDA_AVAILABLE && x isa CuArray

include("Common/common.jl")
include("Clenshaw/ClenshawMapping.jl")
include("Lanczos/Lanczos.jl")

import .Common: Formatter, fmt, bold, cyan, blue, green, red, yellow, @addtime!
import .ClenshawMapping: Clenshaw
import .Lanczos: lanczos, FactorizationReport, display_factorization_report,
        FullRO, PartialRO, ReOrthTechnique,
        MatrixBasis, HybridMatrixBasis, OrthonormalBasis

const main_module_file = abspath(@__FILE__)

include("Polfed/PolfedCore.jl")
using .PolfedCore: polfed, display_report, MappingConfig, TransformConfig, DoSConfig, FactorizationConfig,
    MulColsParallel, TwoLevelParallel, NoParallel, PolfedDefaults, CPU, GPU,
    optimized_mapping!, optimized_clenshaw_recurrence_relation!, optimized_clenshaw_final_sum!
export polfed, display_report
export MappingConfig, TransformConfig, DoSConfig, FactorizationConfig
export MulColsParallel, TwoLevelParallel, NoParallel
export CPU, GPU
export optimized_mapping!, optimized_clenshaw_recurrence_relation!, optimized_clenshaw_final_sum!

end # module
