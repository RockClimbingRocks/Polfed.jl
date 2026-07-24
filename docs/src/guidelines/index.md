# [Guidelines](@id guidlines)

This page collects practical setup rules that are repeatedly referenced in
tutorials.

## Parallelization and Block Size

- Keep `block_size <= 16` in most runs.
  Larger blocks can increase mapping parallelism, but factorization and
  diagonalization costs can grow too much.
- A useful scaling regime is:

```math
\frac{\mathrm{howmany}}{\mathrm{block\_size}} \gtrsim 100.
```

In this regime, column-level mapping parallelization is usually efficient.

## Matrix Size Heuristic

- Around matrix size `~250_000`, [`MulColsParallel`](@ref UniPolfed.MulColsParallel)
  is often the best default strategy on CPU.
- For larger matrices, memory pressure and runtime increase significantly.
  This is where [`TwoLevelParallel`](@ref UniPolfed.TwoLevelParallel) often
  becomes useful, especially with Block Lanczos.

## `howmany` vs Memory Tradeoff

Increasing `howmany` can improve robustness and reduce restart pressure, but
it also raises memory usage and can make projected-eigensolver steps more
expensive.

Use `display(result)` to monitor matrix multiplications, factorization
progress, and the timing split. Inspect
[`order_safety_factor`](@ref UniPolfed.order_safety_factor)
while tuning.
*
