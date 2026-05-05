# # [Reporting](@id tutorial_reporting)
#
# Reporting helps you understand what happened during a run and where time was
# spent.
#
# ## Enable Reporting
#
# To enable reporting, set
# [`produce_report`](@ref Polfed.PolfedCore.PolfedDefaults.produce_report) to
# `true` in the [`polfed`](@ref Polfed.polfed) call:
#
# ```julia
# vals, vecs, report = polfed(mat, x0, howmany, target; produce_report=true)
# display_report(report)
# ```
#
# ## Example Report
#
# A concrete example is often the easiest way to read the report. For example,
# with a small QSun test problem:
#
# ```julia
# using Polfed
# using Polfed.QSun: qsun_hamiltonian
# using LinearAlgebra
# using Random
#
# rng = MersenneTwister(1234)
#
# L_loc = 6
# L_grain = 2
# g0 = 1.0
# α = 0.5
# mat = qsun_hamiltonian(L_loc, L_grain, g0, α; rng=rng, use_sparse=true)
#
# x0 = rand(rng, size(mat, 1))
# x0 ./= norm(x0)
#
# howmany = 20
# target = :middle
#
# vals, vecs, report = polfed(mat, x0, howmany, target; produce_report=true)
# display_report(report)
# ```
#
# one obtains a report of the form
#
# ```text
# Spectral Transformation Report:
# - Targeted 20 eigenpairs with strategy :middle at rescaled energy 0.000000
# - Exposing ev's in the rescaled interval [-3.98e-02, 3.98e-02], with width δ = 7.97e-02
# - Performing 'Chebyshev' spectral transformation of order K = 66 (and order safety factor 0.97)
# - Matrix multiplication performed 4_376 times! With parallelization strategy: MulColsParallel(1)
# - Automatic optimization off
# Factorization Report: (LanczosFactorization with blocksize 1)
# - Number of converged eigenpairs:   20 (out of 20 requested)
# - Lanczos convergence satisfied by: 20 (with tolerance 1.00e-14, max residual 1.84e-16)
# - Eigen convergence satisfied by:   20 (with tolerance 1.00e-09, max residual 1.28e-13)
# - Iterations needed: 66 (out of 68 reserved, overestimated by 2.94%)
# Timings: Percentages are distributed as: (Mapping, Reorthogonalization, Convergence check, others)
# - Total polfed run took: 4.03 seconds (walltime), 3.98 seconds (CPU time)
# - Walltime of factorization took: 0.01 seconds (84.43%, 2.37%, 10.89%, 2.30%)
# - CPU time of factorization took: 0.01 seconds (85.38%, 2.16%, 10.80%, 1.66%)
# ```
#
# The exact values will of course change with the Hamiltonian, random seed,
# `howmany`, target choice, and hardware. But the structure of the printed
# report stays the same. In an interactive terminal,
# [`display_report`](@ref Polfed.display_report) uses colors by default. If you
# redirect the report to a `.txt` file, it is useful to call
# `display_report(report; use_colors=false)` to avoid ANSI color codes.
#
# ## What [`display_report`](@ref Polfed.display_report) Shows
#
# - Spectral transformation summary:
#   target strategy, targeted rescaled energy, targeted interval, interval
#   width, polynomial type, polynomial order,
#   [`order_safety_factor`](@ref Polfed.PolfedCore.PolfedDefaults.order_safety_factor),
#   matrix multiplication count, parallel strategy, and whether automatic
#   optimization was enabled
#   ([`SpectralTransformReport`](@ref Polfed.PolfedCore.SpectralTransformReport)).
# - Factorization summary:
#   Lanczos/Block Lanczos type, block size, converged eigenpair counts, Lanczos
#   and eigen residuals, and reserved vs used iterations
#   ([`FactorizationReport`](@ref Polfed.Lanczos.FactorizationReport)).
#
# The difference between standard Lanczos and Block Lanczos is explained in
# [Lanczos and Block Lanczos Factorization](@ref tutorial_lanczos_block).
# - Timing summary:
#   walltime/CPU-time breakdown across mapping, reorthogonalization, convergence
#   checking, and other operations
#   ([`BenchmarkReport`](@ref Polfed.PolfedCore.BenchmarkReport)).
# - Optional convergence table:
#   per-check evolution of Krylov dimension, converged count, and residual.
# - Optional benchmark details:
#   full-run timing summary and optional memory reporting.
#
# ## Selective Report Display
#
# ```julia
# display_report(
#     report;
#     include_spectral_transform=true,
#     include_factorization=true,
#     include_benchmark=true,
#     show_convergence_details=false,
# )
# ```
#
# - `include_spectral_transform=true`: show spectral transformation block.
# - `include_factorization=true`: show factorization block.
# - `include_benchmark=true`: show benchmark/timing block.
# - `show_convergence_details=false`: print per-check convergence table only
#   when needed. Set it to `true` when studying convergence behavior or when a
#   run does not converge and you want to identify why.
#
# ## Logging Levels
#
# Polfed.jl also offers different logging levels to better understand what is
# going on and if and when does the problem occur.
#
# ```julia
# using Polfed
#
# Polfed.Common.verbosity[] = 0 # silent 
# Polfed.Common.verbosity[] = 1 # warn
# Polfed.Common.verbosity[] = 2 # info
# Polfed.Common.verbosity[] = 3 # debug
# ```
#
# These levels represent the following:
#
# - `0` ([`POLFED_SILENT_LEVEL`](@ref Polfed.Common.POLFED_SILENT_LEVEL)):
#   no logging.
# - `1` ([`POLFED_WARN_LEVEL`](@ref Polfed.Common.POLFED_WARN_LEVEL)):
#   WARN messages only (potential issues / suboptimal setup).
# - `2` ([`POLFED_INFO_LEVEL`](@ref Polfed.Common.POLFED_INFO_LEVEL)):
#   INFO + WARN (major run phases and setup diagnostics).
# - `3` ([`POLFED_DEBUG_LEVEL`](@ref Polfed.Common.POLFED_DEBUG_LEVEL)):
#   DEBUG + INFO + WARN (most detailed tracing, including debug-tagged
#   progress events).
#
# For full report/logging reference, see
# [Reports, Logging, and Defaults](@ref docs_report_defaults).
#
# Use higher levels when diagnosing convergence/setup issues.
