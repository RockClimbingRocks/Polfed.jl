# [Full POLFED](@id tutorial_full)

The whole spectrum — a matrix-free stand-in for dense exact diagonalization. It shares
[Patched POLFED](@ref tutorial_patched)'s edge-first machinery but is its own entrypoint
with its own defaults, because "everything" is not just a very wide window: there is no
target to resolve, and the result is dense by construction.

```julia
using UniPolfed

vals, vecs, report = full_polfed(H, x0, 2000)
```

There is no `target` argument. The rescaled spectrum ``[-1,1]`` is resolved edge-first and
tiled; `howmany_per_patch` is the only sizing knob.

## What you get back

A [`PatchedResult`](@ref UniPolfed.PatchedResult), unpacking as `(vals, vecs, report)` like
every other entrypoint. Here `vecs` is **one dense ``D \times D`` matrix** — the honest cost of
asking for every eigenvector, and what bounds the problem size: full POLFED is memory-limited in
a way that `polfed` and `patched_polfed` are not.

If completeness cannot be certified, the converged pairs are returned **with a warning**
rather than an exception. Check `length(vals)` against `size(H, 1)`.

## Its own defaults

`full_polfed` reads from `Defaults.full_polfed`, not `Defaults.polfed`. Notably
`optimize_mapping` defaults on, the DoS settings are its own, and `retry_budget_growth` is
`1.5` (against `2.0` for `patched_polfed`) — with hundreds of patches, a gentler retry
growth is the better trade. Override anything through the usual keywords:

```julia
vals, vecs, report = full_polfed(H, x0, 2000;
    mc = MappingConfig(optimize_mapping = true),
    dosc = DoSConfig(N = 400, R = 500),
    eigentol = 1e-9,
    edge_distance = 0.2,
    howmany_pad = 3,
    max_retries = 3,
    retry_growth = 1.5,
)
```

## Benchmarking against ED

For a matrix small enough to diagonalize densely, the comparison is the sharpest correctness
test there is — and the natural way to build confidence before scaling up:

```julia
using LinearAlgebra

vals_polfed, _, report = full_polfed(H, x0, 100)
vals_ed = eigvals(Symmetric(Matrix(H)))

length(vals_polfed) == length(vals_ed)
maximum(abs, vals_polfed .- vals_ed)      # should sit near eigentol
```

Both are sorted ascending, so the comparison is elementwise. If the lengths differ, the run
did not certify completeness — read the coverage lines in the report before comparing
values.

Keep the test matrix genuinely small (a few thousand at most): dense `eigvals` is
``\mathcal{O}(\mathcal{D}^3)`` and the dense eigenvector matrix is what you were trying to
avoid in the first place.

!!! warning "Avoid symmetric spectra with centered targets"
        A perfectly symmetric spectrum combined with a centered target produces exact
        degeneracies, which Lanczos cannot separate — you will see `converged = 0` and it
        will look like a bug. Use an asymmetric test Hamiltonian.

## Reading the output

The report is the same [`PatchedReport`](@ref UniPolfed.PatchedReport) that
`patched_polfed` returns, so
[that page's output section](@ref tutorial_patched) applies in full. Three things matter
more here:

- **Decomposition → run count.** With the whole spectrum tiled this is large. Multiply by
  the per-run overhead to sanity-check whether `howmany_per_patch` is too small.
- **Coverage.** In a full run, coverage *is* completeness. Any patch that failed to bracket
  its interval means missing eigenvalues, and the count will not match ``\mathcal{D}``.
- **Aggregate cost** ``\sum_p K_p \cdot bs_p \cdot iters_p``. Compare it against the
  ``\mathcal{O}(\mathcal{D}^3)`` of dense ED to see where the crossover sits for your model.

```julia
display_report(report)                              # the four sections
display_report(report; convergence_details = true)  # + the per-run table
```

## Next

- [Benchmarks](@ref benchmarks) — POLFED versus ED, measured.
- [Going faster](@ref going_faster) — the full run is where kernel and GPU work pays off most.
