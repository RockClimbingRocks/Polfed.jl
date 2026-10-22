# Optimization slice — fast, value-grouped spectral-transform kernels.
#
# The POLFED bottleneck is the K sparse matvecs of H inside every Clenshaw step, so a faster
# `H·v` multiplies through the whole run. For a structured Hamiltonian a purpose-built row kernel
# beats the generic sparse matvec severalfold.
#
# Three layers, deliberately separate:
#
#   MappingStructure.jl   classify(H) → MappingStructure   the FACTS about H (derived)
#   MappingKernel.jl      MappingKernel(...)               the CHOICE of kernel (yours)
#   registry.jl           which kernels exist and what they accept
#
# and under patterns/, ONE FOLDER PER off-diagonal pattern, each self-contained: its kernels, how
# to build them, and whether they may be used at all. Adding a pattern is adding a folder.
#
# The structure is value-grouped — one group per distinct off-diagonal value, each carrying its
# own per-row extents — which is what lets the kernels handle several off-diagonal values AND
# irregular row lengths (symmetry sectors) rather than only the single-value, regular-row case.
#
# Today there is one pattern, `universal`, whose kernels run on every structure `classify` can
# produce. Every built kernel is checked against `mul!` before it reaches the solver.
#
# The module is independently testable; POLFED's matrix entrypoint consumes it through the single
# function `build_optimized_mapping`.

module Optimization

using LinearAlgebra: mul!, norm, BLAS
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange
using StaticArrays: SVector
using SIMD: Vec, vload, vstore, shufflevector
using Printf: @printf, @sprintf

include("MappingStructure.jl")      # ValueGroup + MappingStructure + classify + rescale
include("patterns/patterns.jl")     # one folder per pattern: kernels, build, validate
include("registry.jl")              # KernelInfo + REGISTRY + mapping_kernels
include("MappingKernel.jl")         # the user-facing kernel spec + build_kernel dispatch
include("optimized_mapping.jl")     # resolve_mapping_kernel + build_optimized_mapping
include("benchmark.jl")             # benchmark_mapping + best

export ValueGroup, MappingStructure, classify, rescale,
       FlipStructure, n_masks, build_flip_kernel,
       HopStructure, n_axes, build_hop_kernel,
       n_values, is_regular,
       KernelInfo, REGISTRY, mapping_kernels,
       MappingKernel, kernel_name, build_kernel,
       build_range_kernel, range_units, range_alignment, supports_range,
       can_apply, why_not, verify_kernel,
       resolve_mapping_kernel, build_optimized_mapping, RangeKernelOperator,
       benchmark_mapping, MappingBenchmark, best, per_column

end # module Optimization
