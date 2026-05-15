# [Optimized Mapping](@id tutorial_optimized_mapping)

Enable automatic optimization with
[`MappingConfig`](@ref Polfed.MappingConfig)
(`optimize_mapping=true`).

## Why It Helps

POLFED is commonly memory-bound. For Hamiltonians with few distinct
offdiagonal values, reducing memory traffic in mapping can bring large speedup.

Note: the keyword name is `optimize_mapping` in
[`MappingConfig`](@ref Polfed.MappingConfig).

## Typical Use

```julia
mapping = MappingConfig(
    optimize_mapping=true,
    parallel_strategy=MulColsParallel(),
)

vals, vecs, report = polfed(mat, x0, howmany, target;
    produce_report=true,
    mapping=mapping,
)
display_report(report)
```

This workflow uses [`polfed`](@ref Polfed.polfed) with
[`display_report`](@ref Polfed.display_report) for diagnostics.

## What does it do?

With matrix input, POLFED inspects matrix structure and builds optimized
mapping kernels. The central idea is:

- Split diagonal and offdiagonal contributions,
- Compress repeated offdiagonal values,
- Avoid unnecessary memory loads/stores in repeated mapping calls,
- Reuse optimized kernels in spectral transformation and Clenshaw steps.

Since mapping is applied many times during polynomial filtering, even moderate
per-call memory savings accumulate into meaningful end-to-end speedup.

The next section covers additional memory-access reductions with custom and
rescaled mappings. See [Reducing Memory Access](@ref Reducing_Memory_Access).
*

