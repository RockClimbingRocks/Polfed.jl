# Polfed.jl release notes

## v1.0.0

A rewrite of the package, developed as `UniPolfed.jl` and merged back under the
`Polfed` name. Code written for v0.1 needs changes.

### New

- Entrypoints for patched, full-spectrum, and two-window runs: `patched_polfed`,
  `full_polfed`, and `double_polfed`, next to `polfed`.
- Targets can be a number of states around an energy or an energy window. The filter,
  its order, and the Krylov budget are resolved from a KPM density of states.
- `polfed` returns a `PolfedResult`, which destructures as `(vals, vecs, report)` and
  always carries its `Report`.
- Parallelization is chosen with `BlockWise` or `VectorWise`, and is resolved
  automatically from the block size and thread count when not given.

### Removed

- `TransformConfig` and `FactorizationConfig`. The filter and order are resolved
  automatically, and the Lanczos tolerances and budget are the keyword arguments
  `tol`, `eigentol`, and `overestimate_iters`.
- `MulColsParallel`, `TwoLevelParallel`, and `NoParallel`, replaced by `BlockWise` and
  `VectorWise`.
- `optimized_mapping!`, `optimized_clenshaw_recurrence_relation!`, and
  `optimized_clenshaw_final_sum!`, replaced by `classify`, `resolve_mapping_kernel`,
  and `build_optimized_mapping`.
- The `produce_report` keyword. The report is always returned.

The v0.1 code stays in the git history, tagged `v0.1.0`.
