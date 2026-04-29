# # [Parallelization](@id tutorial_parallelization)
#
# Spectral mapping is usually the dominant runtime component in POLFED, so
# choosing `parallel_strategy` in
# [`MappingConfig`](@ref Polfed.MappingConfig) is important.
#
# ## Strategies
#
# - [`NoParallel`](@ref Polfed.NoParallel)
#   Internal mapping parallelization is disabled. In practice this is most
#   common for GPU workflows or fully user-controlled custom mappings.
#
# - [`MulColsParallel`](@ref Polfed.MulColsParallel)
#   Parallelizes mapping across block columns (vectors). This is the default and
#   usually the best first strategy on CPU.
#
# - [`TwoLevelParallel`](@ref Polfed.TwoLevelParallel)
#   Adds worker-level distribution plus per-column threading. This can help for
#   large runs where worker-startup overhead is amortized.
#
# ## Configuration
#
# ```julia
# using Polfed
#
# mapping_mul = MappingConfig(parallel_strategy=MulColsParallel())
# mapping_two = MappingConfig(parallel_strategy=TwoLevelParallel(2))
# mapping_off = MappingConfig(parallel_strategy=NoParallel())
#
# vals, vecs, report = polfed(mat, x0, howmany, target;
#     produce_report=true,
#     mapping=mapping_mul,
# )
# display_report(report)
# ```
#
# This example uses [`MappingConfig`](@ref Polfed.MappingConfig),
# [`polfed`](@ref Polfed.polfed), and
# [`display_report`](@ref Polfed.display_report).
#
# ## Factorization Interaction
#
# The factorization choice itself is explained in
# [Lanczos and Block Lanczos Factorization](@ref tutorial_lanczos_block).
#
# - `x0::AbstractVector` -> Lanczos.
# - `x0::AbstractMatrix` -> Block Lanczos.
#
# Block Lanczos introduces an additional practical level of parallel work
# (across vectors/columns). Because of that, `TwoLevelParallel` is usually most
# meaningful when running Block Lanczos factorization.
#
# ## [Reasoning About Strategy Choice](@id parallelization_reasoning)
# - [`NoParallel`](@ref Polfed.NoParallel) is usually chosen for GPU workflows.
#   On CPU, [`MulColsParallel`](@ref Polfed.MulColsParallel) and
#   [`TwoLevelParallel`](@ref Polfed.TwoLevelParallel) are typically faster.
# - Around matrix size `~250_000`, time and memory are often still in a regime
#   where [`MulColsParallel`](@ref Polfed.MulColsParallel) is the best choice.
# - For larger matrices, both runtime and memory pressure increase. At that
#   point you usually need more CPU resources, and
#   [`TwoLevelParallel`](@ref Polfed.TwoLevelParallel) often becomes the right
#   option, especially if you keep `block_size <= 16`. Keep in mind that two level parallelization is not that straight forward and that it take time to distribute the tasks! That is why this become benefiical only for larger matrices.
# - Keep the ratio `howmany / block_size >= 100` when possible. See
#   [Guidelines](@ref guidlines).
# - You can increase `howmany`, but this also increases memory use and can make
#   projected matrix diagonalization more expensive. See
#   [Guidelines](@ref guidlines).
