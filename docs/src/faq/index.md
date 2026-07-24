# FAQ / Troubleshooting

## I requested many eigenpairs and runtime increased. Is this expected?

Yes. `howmany` affects polynomial/filter and factorization behavior. Check
reporting output to inspect order and iteration balance.

## Why did factorization type change?

It depends on `x0`:

- vector `x0` -> Lanczos
- matrix `x0` -> Block Lanczos

## Which mapping keyword should I use?

Use `parallel_strategy` inside [`MappingConfig`](@ref UniPolfed.MappingConfig).

## My run is slow. What should I test first?

1. Inspect the returned `result.report`, or call `display(result)`, to see
   mapping/factorization timings.
2. Try `optimize_mapping=true` in
   [`MappingConfig`](@ref UniPolfed.MappingConfig).
3. For structured models, build custom mapping and optionally custom rescaled/Clenshaw kernels.

## How do I debug setup issues?

Increase verbosity:

```julia
UniPolfed.Common.verbosity[] = 3
```
*
