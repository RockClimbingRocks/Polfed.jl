# # [Lanczos and Block Lanczos Factorization](@id tutorial_lanczos_block)
#
# The same [`polfed`](@ref Polfed.polfed) interface supports both Lanczos and
# Block Lanczos factorization. The distinction is determined automatically from
# the shape of the initial input `x0`: a vector selects standard Lanczos, while
# a matrix activates Block Lanczos with block size equal to the number of
# columns.
#
# A minimal example is:
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
# mat = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
# howmany = 100
# target = 0.0
#
# # Vector input -> Lanczos factorization
# x0_vec = rand(size(mat, 1))
# x0_vec ./= norm(x0_vec)
# vals_l, vecs_l = polfed(mat, x0_vec, howmany, target)
#
# # Matrix input -> Block Lanczos factorization
# x0_mat = rand(size(mat, 1), 4)
# x0_mat = Matrix(qr(x0_mat).Q)
# vals_b, vecs_b = polfed(mat, x0_mat, howmany, target)
# ```
#
# In the first call, the input is a single normalized vector, so the algorithm
# uses the standard Lanczos factorization. In the second call, the input is a
# matrix with four columns, so Block Lanczos is used with block size `4`. The
# QR factorization is employed to construct a block of mutually orthonormal
# starting vectors, which is the recommended way to initialize a block run.
#
# Standard Lanczos is usually the more efficient factorization in terms of FLOPs
# and often the better default for smaller runs. Block Lanczos is usually
# worthwhile when the gain from mapping several vectors at once outweighs the
# additional factorization cost.
#
# ## Why Block Lanczos Can Help
#
# Block Lanczos applies the polynomial filter to several vectors
# simultaneously, which improves hardware utilization and is therefore
# especially beneficial for parallel execution on multi-core CPUs and in later
# GPU workflows.
#
# This is one of the main reasons why factorization choice and parallelization
# strategy are closely connected. The factorization decides whether POLFED is
# mapping one vector at a time or an entire block of vectors.
#
# ## Choosing a Reasonable Block Size
#
# A larger block size is not always better. Increasing block size can improve
# parallel efficiency by exposing more independent work during the mapping step,
# but it also changes the cost and memory footprint of the factorization
# itself.
#
# A useful rule of thumb is to choose the block size so that
#
# ```math
# \frac{\texttt{howmany}}{\texttt{block\_size}} \gtrsim 100.
# ```
#
# If this ratio becomes too small, the block factorization typically requires
# many more Krylov iterations to converge, which can make the overall
# computation substantially slower. See [Guidelines](@ref guidlines) for the
# matching practical tuning rules.
#
# ## Krylov Dimension Estimate
#
# This is consistent with the internal estimate for the required Krylov
# dimension,
#
# ```julia
# expectedkrylovdim(howmany::Int, blocksize::Int, η::Real) =
#     ceil(Int64, (20.427 * blocksize + 1.696 * howmany) * η)
# ```
#
# where `η` is an overestimation factor. In the code this role is played by
# [`overestimate_iters`](@ref Polfed.PolfedCore.PolfedDefaults.overestimate_iters),
# and the corresponding helper is
# [`expectedkrylovdim`](@ref Polfed.PolfedCore.PolfedDefaults.expectedkrylovdim).
#
# This estimate grows linearly with `blocksize`, so choosing an unnecessarily
# large block can significantly increase the size of the projected problem and
# slow down convergence rather than improving it.
#
# ## `howmany` vs Memory Tradeoff
#
# Increasing `howmany` can, on the other hand, be beneficial. Requesting more
# eigenpairs often decreases the required polynomial order, which reduces the
# cost of the spectral transformation and can lead to a net speedup.
#
# However, `howmany` cannot be increased indefinitely. A larger value also leads
# to a larger Krylov subspace, which increases both the memory needed to store
# the Krylov vectors, usually the dominant memory cost of the algorithm, and the
# size of the projected Lanczos matrix. Beyond some point, memory requirements
# become prohibitive, and the cost of diagonalizing the projected matrix can
# outweigh the benefit of the lower polynomial order.
#
# In practice, good performance is obtained by balancing these competing effects
# rather than maximizing either `block_size` or `howmany` in isolation.
#
# ## Reporting
#
# When reporting is enabled, the factorization summary indicates which mode was
# used together with the block size and iteration count, making it
# straightforward to compare different configurations.
#
# ```julia
# vals, vecs, report = polfed(mat, x0_mat, howmany, target; produce_report=true)
# display_report(report)
# ```
#
# This is the most convenient way to compare Lanczos and Block Lanczos runs in
# practice before tuning parallelization.
