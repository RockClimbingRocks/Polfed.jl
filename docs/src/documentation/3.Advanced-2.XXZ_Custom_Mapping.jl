# # [Custom Mapping](@id tutorial_xxz_custom_mapping)
#
# This section shows how to exploit XXZ structure explicitly.
# See [Optimization of the XXZ Model](@ref optimization_xxz_model), where the
# XXZ model, optimization logic, and helper functions are defined.
#
# We use the fact that many offdiagonal elements share one value and that the
# mapping kernel is memory-bound. By reducing index/data traffic in mapping, we
# usually speed up the full POLFED pipeline.
#
# ## Build a Structured Mapping
#
# The following helpers are defined in
# [Optimization of the XXZ Model](@ref optimization_xxz_model):
#
# - `construct_XXZ_matrix(L, Delta, Nup)`
# - `get_diags_and_offdiagonals_single_value(Delta, L, Nup; ...)`
# - `mapvec_with_xxz!(...)`
#
# In practice, we first construct diagonal/offdiagonal elements and then use
# them to construct a custom mapping.
#
# ```julia
# using Polfed
# using BenchmarkTools
# using LinearAlgebra
#
# L = 18
# Nup = L ÷ 2
# Delta = 0.55
#
# mat = construct_XXZ_matrix(L, Delta, Nup)
# diags, offdiag_val, flat, starts = get_diags_and_offdiagonals_single_value(Delta, L, Nup)
# f_custom! = mapvec_with_xxz!(diags, offdiag_val, flat, starts)
# ```
#
# What these two helpers do:
#
# - `get_diags_and_offdiagonals_single_value(Delta, L, Nup; ...)`
#   builds the XXZ matrix from model parameters and extracts a compact
#   representation: diagonal values (`diags`), the common offdiagonal value
#   (`offdiag_val`), flattened offdiagonal column indices (`flat`), and
#   row-start pointers (`starts`).
#
# - `mapvec_with_xxz!(diags, offdiag_val, flat, starts)`
#   builds and returns a mapping function `f_custom!(Y, X)` that applies the
#   matrix action using that compact representation, avoiding generic sparse
#   matrix traversal overhead in each mapping call.
#
# For more details about the XXZ model and the related mapping, see
# [Optimization of the XXZ Model](@ref optimization_xxz_model).
#
# ## Benchmark Mapping Against Generic `mul!`
#
# Spectral transformation is the dominant runtime component of POLFED. Because
# of that, optimizing mapping often speeds up almost the whole algorithm and
# decreases runtime.
#
# When constructing new mappings, it is useful to benchmark mapping kernels
# directly first. There are multiple reasons. First, this isolates benchmarks
# to mapping only, making them fast and efficient to test. This also avoids
# getting different results from different `polfed` runs due to different
# memory allocation/layout effects. Benchmarking only the mapping repeatedly
# evaluates the same kernel, so the average time is reliable. In the end, this
# tells you whether speedup is expected for this specific mapping.
#
# ```julia
# X = rand(size(mat, 1)); Y = similar(X)
# @btime mul!($Y, $mat, $X)
# @btime $f_custom!($Y, $X)
# ```
#
# For smaller system sizes (`L=18`) there is an approximate speedup factor of
# `~1.4`, whereas for larger system sizes a factor of `~2.5` up to `~3.0` seems
# to persist.
#
# ```@raw html
# <table>
#   <thead>
#     <tr>
#       <th style="text-align:center;">L</th>
#       <th style="text-align:center;">generic <code>mul!</code></th>
#       <th style="text-align:center;">custom mapping</th>
#       <th style="text-align:center;">speedup factor</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td style="text-align:right;">18</td>
#       <td style="text-align:right;">856.653 μs</td>
#       <td style="text-align:right;">579.533 μs</td>
#       <td style="text-align:right;">1.48</td>
#     </tr>
#     <tr>
#       <td style="text-align:right;">20</td>
#       <td style="text-align:right;">5.402 ms</td>
#       <td style="text-align:right;">2.340 ms</td>
#       <td style="text-align:right;">2.31</td>
#     </tr>
#     <tr>
#       <td style="text-align:right;">22</td>
#       <td style="text-align:right;">25.727 ms</td>
#       <td style="text-align:right;">10.202 ms</td>
#       <td style="text-align:right;">2.52</td>
#     </tr>
#     <tr>
#       <td style="text-align:right;">24</td>
#       <td style="text-align:right;">109.943 ms</td>
#       <td style="text-align:right;">41.825 ms</td>
#       <td style="text-align:right;">2.63</td>
#     </tr>
#   </tbody>
# </table>
# ```
#
# ## Use Custom Mapping in POLFED
#
# After the mapping benchmark is finished, we can test it in full POLFED runs.
#
# ```julia
# x0 = rand(size(mat, 1)); x0 ./= norm(x0)
# howmany = 120
# target = 0.0
#
# f_mul! = (Y, X) -> mul!(Y, mat, X)
#
# # Full POLFED with matrix -> mul! path
# vals_mul, vecs_mul, report_mul = polfed(mat, x0, howmany, target; produce_report=true)
#
# # Full POLFED with custom mapping path
# vals_custom, vecs_custom, report_custom = polfed(f_custom!, x0, howmany, target; produce_report=true)
#
# display_report(report_mul)
# display_report(report_custom)
# ```
#
# The speedup trend should be similar to the mapping-kernel benchmark.
# If `f_custom!` is faster than `mul!`, POLFED usually benefits similarly
# because mapping dominates total work. Compare matrix multiplications and
# timing blocks in the two reports.
#
# As discussed in [Optimized Mapping](@ref tutorial_optimized_mapping) and
# [Reducing Memory Access](@ref Reducing_Memory_Access), it is important to
# reduce memory access as much as possible. That is why it is beneficial to
# construct a rescaled mapping.
#
# ```julia
# Emin = first(collect(Polfed.Lanczos.lanczos(f_custom!, x0, 1; which=:SR, maxdim=1000)[1]))
# Emax = last(collect(Polfed.Lanczos.lanczos(f_custom!, x0, 1; which=:LR, maxdim=1000)[1]))
# spread = (Emax - Emin) / 2
# center = (Emax + Emin) / 2
#
# diags_rescaled = @. (diags - center) / spread
# offdiag_val_rescaled = offdiag_val * (1 / spread)
# f_benchmark_rescaled! = mapvec_with_xxz!(diags_rescaled, offdiag_val_rescaled, flat, starts)
#
# mapping_rescaled = MappingConfig(
#     f!_rescaled=f_benchmark_rescaled!,
#     Emin=Emin,
#     Emax=Emax,
# )
# vals_rescaled, vecs_rescaled, report_rescaled = polfed(f_custom!, x0, howmany, target; produce_report=true, mapping=mapping_rescaled)
# display_report(report_rescaled)
# ```
#
# This gives another speedup by reducing runtime rescaling overhead.
# This is exactly what POLFED does for you when `optimize_mapping=true` is used,
# as demonstrated in [Automatic Optimization](@ref tutorial_xxz_baseline).
