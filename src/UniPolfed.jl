module UniPolfed

# Rebuilt incrementally on the `restructure/file-layout` branch. Old code lives
# in ../src-OBSOLET as the reference we port from. Components are migrated
# dependency-first: infrastructure → Numerics (Clenshaw → KPM → Lanczos) → POLFED.

using LinearAlgebra, Random

# ── infrastructure (migrated) ────────────────────────────────────────────────
include("common/Common.jl")    # logging, formatting, timing            (module Common)
include("backend/Backend.jl")  # GPU dispatch plumbing + CPU/GPU types   (top-level)

# ── numerics (migrated: Clenshaw) ─────────────────────────────────────────────
include("numerics/Numerics.jl") # reusable spectral primitives           (module Numerics)

# ── physics (migrated: Models) ────────────────────────────────────────────────
include("models/Models.jl")     # Hamiltonian builders (QSun, spin chains) (module Models)

# ── optimization (migrating: Stage 1 — reads kernels + benchmark) ─────────────
include("optimization/Optimization.jl")  # value-grouped row kernels        (module Optimization)

# ── POLFED core (migrating) ───────────────────────────────────────────────────
using Distributed                                             # TwoLevelParallel workers
using SharedArrays                                            # TwoLevelParallel Clenshaw buffer
using Printf                                                  # Report section formatting
using PrettyTables                                            # Report convergence-history table
using LinearAlgebra: BLAS                                     # BLAS-thread gating in clenshaw_apply
using .Common: should_log, polfed_log,                        # gated POLFED logging
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
                         Layout, ColumnLayout, RowLayout,      # Krylov block layout (row-native path)
                         LanczosResult, lanczos,               # entrypoint factorization
                         lanczos_extrema,                      # Mapping bound estimation
                         FactorizationReport                   # Report convergence/timing source
using .Optimization: build_optimized_mapping                  # matrix entrypoint row kernels

include("main/main.jl")                          # module Defaults (filter-agnostic, algorithm-wide)




end # module
