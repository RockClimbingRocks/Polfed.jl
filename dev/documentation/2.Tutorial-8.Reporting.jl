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
# using Polfed.Models: qsun_hamiltonian
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
# on the webpage, this report can be rendered with the same terminal-style
# color structure:
#
# ```@raw html
# <div style="margin:1rem 0; border:1px solid #3a3a3a; border-radius:8px; background:#171717; overflow-x:auto; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);">
#   <div style="margin:0; padding:0.95rem 1.05rem; font-family:JuliaMono, SFMono-Regular, Menlo, Consolas, Liberation Mono, DejaVu Sans Mono, monospace; font-size:0.92rem; line-height:1.5; color:#e8e8e8; white-space:pre;"><span style="font-weight:700; color:#5f87ff;">Spectral Transformation Report:</span>
# - Targeted <span style="font-weight:700; color:#5fd7d7;">20</span> eigenpairs with strategy <span style="font-weight:700; color:#5fd7d7;">:middle</span> at rescaled energy <span style="font-weight:700; color:#5fd7d7;">0.000000</span>
# - Exposing ev's in the rescaled interval [<span style="font-weight:700; color:#5fd7d7;">-3.98e-02</span>, <span style="font-weight:700; color:#5fd7d7;">3.98e-02</span>], with width <span style="font-weight:700; color:#5fd7d7;">δ = 7.97e-02</span>
# - Performing '<span style="font-weight:700; color:#5fd7d7;">Chebyshev</span>' spectral transformation of order <span style="font-weight:700; color:#5fd7d7;">K = 66</span> (and order safety factor <span style="font-weight:700; color:#5fd7d7;">0.97</span>)
# - Matrix multiplication performed <span style="font-weight:700; color:#5fd7d7;">4_376</span> times! With parallelization strategy: <span style="font-weight:700; color:#5fd7d7;">MulColsParallel(1)</span>
# - Automatic optimization <span style="font-weight:700; color:#ffd75f;">off</span>
# <span style="font-weight:700; color:#5f87ff;">Factorization Report:</span> (<span style="font-weight:700; color:#f5f5f5;">LanczosFactorization</span> with blocksize <span style="font-weight:700; color:#f5f5f5;">1</span>)
# - Number of converged eigenpairs:   <span style="font-weight:700; color:#5fd75f;">20</span> (out of <span style="font-weight:700; color:#5fd7d7;">20</span> requested)
# - Lanczos convergence satisfied by: <span style="font-weight:700; color:#5fd75f;">20</span> (with tolerance <span style="font-weight:700; color:#5fd7d7;">1.00e-14</span>, max residual <span style="font-weight:700; color:#ffd75f;">1.84e-16</span>)
# - Eigen convergence satisfied by:   <span style="font-weight:700; color:#5fd75f;">20</span> (with tolerance <span style="font-weight:700; color:#5fd7d7;">1.00e-09</span>, max residual <span style="font-weight:700; color:#ffd75f;">1.28e-13</span>)
# - Iterations needed: <span style="font-weight:700; color:#f5f5f5;">66</span> (out of <span style="font-weight:700; color:#5fd7d7;">68</span> reserved, overestimated by <span style="font-weight:700; color:#ff5f5f;">2.94%</span>)
# <span style="font-weight:700; color:#5f87ff;">Timings:</span> Percentages are distributed as: (Mapping, Reorthogonalization, Convergence check, others)
# - Total polfed run took: <span style="font-weight:700; color:#5fd7d7;">3.89 seconds</span> (walltime), <span style="font-weight:700; color:#5fd7d7;">3.88 seconds</span> (CPU time)
# - Walltime of factorization took: <span style="font-weight:700; color:#5fd7d7;">0.01 seconds</span> (<span style="font-weight:700; color:#5fd7d7;">84.14%</span>, <span style="font-weight:700; color:#5fd7d7;">2.53%</span>, <span style="font-weight:700; color:#5fd7d7;">10.99%</span>, <span style="font-weight:700; color:#5fd7d7;">2.34%</span>)
# - CPU time of factorization took: <span style="font-weight:700; color:#5fd7d7;">0.01 seconds</span> (<span style="font-weight:700; color:#5fd7d7;">85.05%</span>, <span style="font-weight:700; color:#5fd7d7;">2.35%</span>, <span style="font-weight:700; color:#5fd7d7;">10.82%</span>, <span style="font-weight:700; color:#5fd7d7;">1.78%</span>)</div>
# </div>
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
