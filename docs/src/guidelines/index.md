# [Guidelines](@id guidelines)

Which variant, which settings. These are practical rules of thumb, not theorems — the report
from an actual run always beats them.

## Which variant?

```text
Do you want the whole spectrum?
├─ yes ────────────────────────────────────► full_polfed
└─ no
   How many eigenpairs?
   ├─ up to a few thousand ────────────────► polfed
   └─ many thousands (≳ 5000) ─────────────► patched_polfed
```

| Variant | Use when | Returns |
| :--- | :--- | :--- |
| [`polfed`](@ref tutorial_polfed) | one target, one window; the default | [`PolfedResult`](@ref UniPolfed.PolfedResult) |
| [`patched_polfed`](@ref tutorial_patched) | `howmany` is large enough that the dense diagonalization of the projected matrix, or the Krylov basis in memory, dominates the run | [`PatchedResult`](@ref UniPolfed.PatchedResult) |
| [`full_polfed`](@ref tutorial_full) | you want everything, and ``D \times D`` eigenvectors fit in memory | [`PatchedResult`](@ref UniPolfed.PatchedResult) |

The signal to switch from `polfed` to `patched_polfed` is **not** a large polynomial order —
a big `howmany` gives you a *small* order. It is the *Timings* section: when the time spent
diagonalizing the projected Lanczos matrix starts to rival the time spent on matvecs, or when
the Krylov basis no longer fits comfortably in memory, split the request into several smaller
runs. See [The other bottleneck](@ref going_faster).

## Block size and `howmany`

```math
\frac{\texttt{howmany}}{\texttt{block\_size}} \gtrsim 100,
\qquad \texttt{block\_size} \le 16 .
```

Below that ratio, block factorization typically needs many more Krylov iterations, which
erases the mapping gain. The estimated Krylov dimension grows **linearly** in the block
size, so an oversized block enlarges the projected problem for nothing.

Raising `howmany` cuts both ways:

| Raising `howmany` | Effect |
| :--- | :--- |
| widens the target interval | **lowers** the polynomial order ``K`` — often a net speedup |
| enlarges the Krylov basis | **raises** memory, usually the dominant cost |
| enlarges the projected matrix | **raises** the cost of diagonalizing it |

There is an optimum in the middle. Increase `howmany` while the order keeps falling and
memory allows; stop when the Lanczos time stops improving.

## Parallel strategy by problem size

| Situation | Strategy |
| :--- | :--- |
| GPU arrays | [`BlockWise`](@ref UniPolfed.BlockWise) — the device parallelizes itself |
| Single vector | [`BlockWise`](@ref UniPolfed.BlockWise) — the only one that can split one vector |
| Block Lanczos, `block_size == nthreads()` | [`VectorWise`](@ref UniPolfed.VectorWise) — one vector per thread |
| Block Lanczos, any other block size | [`BlockWise`](@ref UniPolfed.BlockWise) |

Both use every thread; they differ in where the parallel region opens — inside each Clenshaw
step (`BlockWise`) or around the whole recurrence (`VectorWise`). Leaving `parallel_strategy`
unset resolves this table for you. Details in [Parallelization](@ref parallelization).

## Targets

- Prefer `:maxdos` when you want "the middle of the spectrum" in the physically meaningful
  sense; prefer `:middle` or `(:rescaled, 0.0)` when you want the geometric center.
- `(:offset, η)` and `(:rescaled, ε)` measure from **different origins**. See
  [Target modes](@ref polfed_targets).
- If a `:maxdos` result surprises you, raise `N` and `R` in
  [`DoSConfig`](@ref UniPolfed.DoSConfig) and check whether the detected peak moved.
- **Avoid a symmetric spectrum with a centered target** in tests. Exact degeneracies cannot
  be separated by Lanczos, and you will see `converged = 0` for reasons that have nothing to
  do with your settings.

## Tolerances

| Keyword | Default | Meaning |
| :--- | :--- | :--- |
| `tol` | `1e-14` | Lanczos breakdown / Ritz tolerance |
| `eigentol` | `1e-9` | acceptance: ``\lVert H v - \lambda v\rVert < \texttt{eigentol}`` |
| `overestimate_iters` | `1.25` | Krylov budget inflation |

`eigentol` is the one with physical meaning. Tightening it beyond what your element type
supports guarantees non-convergence — in `Float32`/`ComplexF32`, `1e-9` is unreachable.

Raise `overestimate_iters` when runs exhaust their budget just short of converging; it is
cheaper than restarting with a larger `howmany`.

## Speed, in order of effort

1. `optimize_mapping = true` if the matrix is structured — one keyword.
2. Check block size and parallel strategy against the tables above.
3. Move to GPU if you have one.
4. Write a structured kernel, and supply `f!_rescaled` with `Emin`/`Emax`.
5. If `howmany` is large enough that the projected diagonalization or the Krylov basis
   dominates, switch to `patched_polfed`.

Full reasoning in [Going faster](@ref going_faster).

## Always look at the report

```julia
result = polfed(H, x0, howmany, target)
display_report(result)
```

Polynomial order, matrix-multiplication count, converged counts, residuals and the timing
split are all there. When comparing two configurations, the **matrix-multiplication count
should be identical** — if it moved, you changed the problem rather than its speed.
