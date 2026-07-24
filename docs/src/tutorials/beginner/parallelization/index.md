# [Parallelization](@id tutorial_parallelization)

Spectral mapping is usually the dominant runtime component in POLFED, so
choosing `parallel_strategy` in
[`MappingConfig`](@ref UniPolfed.MappingConfig) is important.

## Strategies

- [`NoParallel`](@ref UniPolfed.NoParallel)
  Internal mapping parallelization is disabled. In practice this is most
  common for GPU workflows or fully user-controlled custom mappings.

- [`MulColsParallel`](@ref UniPolfed.MulColsParallel)
  Parallelizes mapping across block columns (vectors). This is the default and
  usually the best first strategy on CPU.

- [`TwoLevelParallel`](@ref UniPolfed.TwoLevelParallel)
  Adds worker-level distribution plus per-column threading. This can help for
  large runs where worker-startup overhead is amortized.

## Configuration

```julia
using UniPolfed

mc_mul = MappingConfig(parallel_strategy=MulColsParallel())
mc_two = MappingConfig(parallel_strategy=TwoLevelParallel(2))
mc_off = MappingConfig(parallel_strategy=NoParallel())

result = polfed(mat, x0, howmany, target; mc=mc_mul)
display(result)
```

This example uses [`MappingConfig`](@ref UniPolfed.MappingConfig),
[`polfed`](@ref UniPolfed.polfed), and
`display(result)`.

## Factorization Interaction

The factorization choice itself is explained in
[Lanczos and Block Lanczos Factorization](@ref tutorial_lanczos_block).

- `x0::AbstractVector` -> Lanczos.
- `x0::AbstractMatrix` -> Block Lanczos.

Block Lanczos introduces an additional practical level of parallel work
(across vectors/columns). Because of that, `TwoLevelParallel` is usually most
meaningful when running Block Lanczos factorization.

## [Reasoning About Strategy Choice](@id parallelization_reasoning)
- [`NoParallel`](@ref UniPolfed.NoParallel) is usually chosen for GPU workflows.
  On CPU, [`MulColsParallel`](@ref UniPolfed.MulColsParallel) and
  [`TwoLevelParallel`](@ref UniPolfed.TwoLevelParallel) are typically faster.
- Around matrix size `~250_000`, time and memory are often still in a regime
  where [`MulColsParallel`](@ref UniPolfed.MulColsParallel) is the best choice.
- For larger matrices, both runtime and memory pressure increase. At that
  point you usually need more CPU resources, and
  [`TwoLevelParallel`](@ref UniPolfed.TwoLevelParallel) often becomes the right
  option, especially if you keep `block_size <= 16`. Keep in mind that two level parallelization is not that straight forward and that it take time to distribute the tasks! That is why this become benefiical only for larger matrices.
- Keep the ratio `howmany / block_size >= 100` when possible. See
  [Guidelines](@ref guidlines).
- You can increase `howmany`, but this also increases memory use and can make
  projected matrix diagonalization more expensive. See
  [Guidelines](@ref guidlines).
*
