# Optimization slice — fast, value-grouped spectral-transform kernels.
#
# The POLFED bottleneck is the K sparse matvecs of H inside every Clenshaw step. For structured
# Hamiltonians with few distinct off-diagonal values, custom row-layout kernels beat the generic
# sparse matvec severalfold. This module holds those kernels, a registry, and a benchmark to pick
# the fastest for a given machine/block size.
#
# It provides the general reads kernels (`csr_simd`, `ell32_svec`, `csr64_svec`),
# the structured `shuf` kernel, value grouping, mapping selection, and benchmarks.
# The module is independently testable; the POLFED matrix entrypoint consumes its
# resolved row operators through `build_optimized_mapping`.

module Optimization

using LinearAlgebra: mul!, norm
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
using StaticArrays: SVector
using SIMD: Vec, vload, vstore, shufflevector
using Printf: @printf, @sprintf

include("value_grouping.jl")            # MappingStructure + value_grouping
include("cpu/kernel_helpers.jl")        # _chunks_aligned + loadv/storev!
include("cpu/reads.jl")                 # csr_simd / ell32_svec / csr64_svec row kernels
include("cpu/flips.jl")                 # shuf σˣ flip kernel (SIMD column-packed shuffles)
include("registry.jl")                  # REGISTRY + build_row_mapping + list_mappings
include("benchmark.jl")                 # benchmark_mappings + MappingBench
include("optimized_mapping.jl")         # rescale + resolve_mapping_kernel + build_optimized_mapping

export MappingStructure, value_grouping,
       build_row_mapping, list_mappings,
       benchmark_mappings, MappingBench,
       rescale, resolve_mapping_kernel, build_optimized_mapping

end # module Optimization
