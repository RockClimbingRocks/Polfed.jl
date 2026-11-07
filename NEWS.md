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
- Every entrypoint runs on a CUDA GPU when `H` (or `f!`) and `x0` are device arrays —
  scalar and block Lanczos, real and complex. Load `CUDA` (≥ 5.8; use 6 on Grace Hopper,
  see below) to activate the extension.
- `MappingConfig(layout = …)` picks the Krylov block layout of the matrix entrypoints:
  `RowLayout()` or `ColumnLayout()`. Automatic by default — row on the CPU; on the GPU
  column below block size 8 (a vector `x0` runs scalar Lanczos) and row from 8 on, where
  CUSPARSE's row-major product is cheapest per vector.

### Fixed

- The matrix entrypoints returned the complex conjugate of the eigenvectors of a complex
  Hermitian `H` (eigenvalues were right). The row layout now applies `H` to every row.

### Requirements

- PrettyTables 3 (was 2). Needed for CUDA.jl 6, whose 5.8 predecessor crashes
  intermittently on Grace Hopper (GH200) nodes when reading scalars back from the device.

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
