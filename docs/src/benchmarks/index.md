# [Benchmarks](@id benchmarks)

!!! warning "Placeholder — no measurements published yet"
    This page is a stub. The numbers below are **not** filled in. Do not cite anything from
    this page until it carries a hardware and version stamp.

Benchmarks are kept separate from [Guidelines](@ref guidelines) deliberately: guidelines are
evergreen advice, benchmarks are measurements that go stale. Every table here must state the
machine, the Julia version, the package version and the matrix dimension, or it is not
reproducible and does not belong on the page.

## Reporting template

Fill this in for every measurement set:

| Field | Value |
| :--- | :--- |
| CPU / GPU | — |
| Cores / threads used | — |
| RAM | — |
| Julia version | — |
| UniPolfed version | — |
| Element type | — |
| Date measured | — |

## Planned content

### Accuracy versus dense ED

For matrices small enough to diagonalize densely, compare eigenvalue-by-eigenvalue against
`eigvals`. This is the correctness benchmark; it should show residuals at the level of
`eigentol`.

```julia
using LinearAlgebra

vals_polfed, _, report = full_polfed(H, x0, howmany_per_patch)
vals_ed = eigvals(Symmetric(Matrix(H)))

maximum(abs, vals_polfed .- vals_ed)
```

- [ ] Max and mean absolute eigenvalue error versus dimension.
- [ ] Residual distribution ``\lVert H v - \lambda v \rVert``.
- [ ] Behaviour in `Float32` / `ComplexF32` versus `Float64` / `ComplexF64`.

### Timing versus dense ED

- [ ] Wall time versus dimension, POLFED and `eigvals`, showing the crossover.
- [ ] The crossover point stated explicitly, per model.
- [ ] Memory high-water mark for both.

### Scaling

- [ ] Time versus `howmany` at fixed dimension — showing the ``K`` trade-off.
- [ ] Time versus `block_size`, confirming the ``\texttt{howmany}/\texttt{block\_size} \gtrsim 100`` rule.
- [ ] Thread scaling for `BlockWise` and for `VectorWise`.
- [ ] `BlockWise` versus `VectorWise` at `block_size == nthreads()`, where auto-resolution
      switches between them.
- [ ] CPU versus GPU on the same problem.

### Variant comparison

- [ ] `polfed` versus `patched_polfed` as the window widens — where patching starts to win.
- [ ] Aggregate cost ``\sum_p K_p \cdot bs_p \cdot iters_p`` versus `howmany_per_patch`.

## Method notes

When these are measured, follow the same discipline the tutorials recommend:

- Benchmark the mapping kernel separately from the full solve — it isolates the variable
  and avoids attributing allocation noise to the wrong place.
- Between two configurations of the *same* problem, the matrix-multiplication count must be
  identical. If it is not, the runs are not comparable.
- Report wall time. CPU time is meaningless for GPU runs and is suppressed there.
- Use an asymmetric spectrum, or an off-center target: a symmetric spectrum with a centered
  target produces exact degeneracies and spurious non-convergence.
