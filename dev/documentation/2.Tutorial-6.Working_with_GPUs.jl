# # Working with GPUs
#
# POLFED spends most runtime in spectral transformation, i.e. repeated
# matrix-vector or matrix-matrix products inside polynomial filtering.
#
# GPUs are often much faster for this workload because:
# - mapping is massively data-parallel,
# - POLFED kernels are largely memory-bound and GPUs provide much larger memory
#   bandwidth,
# - regular access patterns can be executed efficiently by many concurrent
#   threads.
#
# In practical usage the interface is almost unchanged: you move arrays to CUDA,
# then call [`polfed`](@ref Polfed.polfed) the same way as on CPU.
#
# ## 1) Running QSun on CPUs
#
# ```julia
# using Polfed
# using Polfed.Models: qsun_hamiltonian
# using LinearAlgebra
#
# L_loc = 12
# L_grain = 2
# g0 = 1.0
# α = 0.5
# mat_cpu = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
# x0_cpu = rand(size(mat_cpu, 1)); x0_cpu ./= norm(x0_cpu)
#
# howmany = 80
# target = 0.0
#
# vals_cpu, vecs_cpu, report_cpu = polfed(
#     mat_cpu, x0_cpu, howmany, target; produce_report=true
# )
# display_report(report_cpu)
# ```
#
# ## 2) Running QSun on GPUs
#
# ```julia
# using CUDA
#
# mat_gpu = CuArray(Matrix(mat_cpu))
# x0_gpu = CUDA.rand(Float64, size(mat_gpu, 1))
# x0_gpu ./= norm(x0_gpu)
#
# vals_gpu, vecs_gpu, report_gpu = polfed(
#     mat_gpu, x0_gpu, howmany, target; produce_report=true
# )
# display_report(report_gpu)
# ```
#
# The call pattern is basically the same; the main difference is array type
# (`Array` vs `CuArray`). Comparing `report_cpu` and `report_gpu` with
# [`display_report`](@ref Polfed.display_report) usually shows large speedups.
#
# In many workloads this speedup is "almost for free" at the software level:
# once GPU hardware is available, the code path change is small.
#
# More detail on optimization patterns and speedup behavior is covered in the
# advanced tutorials.
