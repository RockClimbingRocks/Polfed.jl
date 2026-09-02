module UniPolfed

# Components are loaded in dependency order: infrastructure, reusable numerics,
# model builders, optimized mappings, and finally the POLFED pipelines.

using LinearAlgebra, Random

# ── infrastructure ───────────────────────────────────────────────────────────
include("common/Common.jl")    # logging, formatting, timing            (module Common)
include("backend/Backend.jl")  # GPU dispatch plumbing + CPU/GPU types   (top-level)

# ── reusable numerics ────────────────────────────────────────────────────────
include("numerics/Numerics.jl") # reusable spectral primitives           (module Numerics)

# ── physics models ───────────────────────────────────────────────────────────
include("models/Models.jl")     # Hamiltonian builders (QSun, spin chains) (module Models)

# ── optimized mappings ───────────────────────────────────────────────────────
include("optimization/Optimization.jl")  # value-grouped row kernels        (module Optimization)

# ── POLFED algorithm ─────────────────────────────────────────────────────────
using Printf                                                  # Report section formatting
using PrettyTables                                            # Report detail tables
using LinearAlgebra: BLAS                                     # BLAS-thread gating in clenshaw_apply
using MKLSparse                                               # MKL sparse matrix multiplication (CPU) for row-native path
using MKL_jll: libmkl_rt                                      # MKL sparse matrix multiplication (CPU) for row-native path
using .Common: should_log, polfed_log, with_verbosity,        # gated POLFED logging
               POLFED_SILENT_LEVEL, POLFED_WARN_LEVEL,        # verbosity levels (warn_parallelization gating)
               POLFED_INFO_LEVEL, POLFED_DEBUG_LEVEL, POLFED_DEEP_DEBUG_LEVEL,
               bisection, integrate_chebyshev_weighted,       # Transform bounds/order resolution
               validate_no_leakage_in_interval_numerically,   # Filter leakage validation
               Formatter, fmt, bold, cyan, blue, green, red, yellow,  # Report coloring
               StageTimer, @stage, @stages,                   # named stage timing (pipelines + Lanczos)
               stage_names, total_wall, total_cpu
using .Numerics.Clenshaw: ClenshawExpansion, build_clenshaw_kernels  # Transform filter evaluation
using .Numerics.KPM: kpm_moments, kpm_density, kpm_coeffs, get_kernel # DoS adapter + Transform state count
using .Numerics.Lanczos: ReOrthTechnique, OrthonormalBasis,   # Factorization config types
                         EigTarget, HighestEigs, LowestEigs, BothEnds,
                         Highest, Lowest, ToValue, Window,     # window-mode eig-targets
                         NumEigs,                              # eig-target rendering (trace)
                         Layout, ColumnLayout, RowLayout,      # Krylov block layout (row-native path)
                         blocksize, opdim,                     # layout-aware block shape accessors
                         LanczosResult, lanczos,               # entrypoint factorization
                         lanczos_extrema,                      # Mapping bound estimation
                         FactorizationReport,                  # Report convergence/timing source
                         CONVERGENCE_TABLE_HEADER,             # shared per-check table (both report sinks)
                         convergence_table_rows
using .Optimization: build_optimized_mapping, MappingKernel,   # matrix entrypoint row kernels
                     kernel_name, mapping_kernels, classify,
                     benchmark_mapping, MappingBenchmark, best, per_column

include("main/main.jl")                          # module Defaults + public pipelines
end # module
