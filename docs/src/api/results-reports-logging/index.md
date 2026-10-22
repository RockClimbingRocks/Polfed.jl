# [Results, Reports, Logging](@id api_results_reports)

Reports are part of every solver result; there is no flag to enable them.

**Every entrypoint returns a result object that destructures as `(vals, vecs, report)`** — the
same protocol as [`lanczos`](@ref UniPolfed.Numerics.Lanczos.lanczos):

| Entrypoint | Returns | Report type |
| :--- | :--- | :--- |
| [`polfed`](@ref UniPolfed.polfed) | [`PolfedResult`](@ref UniPolfed.PolfedResult) | [`Report`](@ref UniPolfed.Report) |
| [`patched_polfed`](@ref UniPolfed.patched_polfed) | [`PatchedResult`](@ref UniPolfed.PatchedResult) | [`PatchedReport`](@ref UniPolfed.PatchedReport) |
| [`full_polfed`](@ref UniPolfed.full_polfed) | [`PatchedResult`](@ref UniPolfed.PatchedResult) | [`PatchedReport`](@ref UniPolfed.PatchedReport) |
| [`double_polfed`](@ref UniPolfed.double_polfed) | `DoublePolfedResult` | two `Report`s |

`double_polfed` is the exception: it has two windows, so it destructures into the two
`PolfedResult`s — each of which then unpacks as `(vals, vecs, report)`.

## Results

### [`PolfedResult`](@id UniPolfed.PolfedResult)

The standard solver result. Fields:

| Field | Contents |
| :--- | :--- |
| `vals` | eigenvalues, ascending, as true energies of ``H`` |
| `vecs` | matching eigenvectors |
| `residuals` | ``\lVert H v - \lambda v \rVert`` per pair |
| `converged` | number of converged pairs |
| `layout` | the Krylov block layout used |
| `report` | the structured [`Report`](@ref UniPolfed.Report) |

`vals`, `vecs` and `residuals` are sorted with respect to ``H``, and the returned subset is
always a contiguous run — so the eigenvector subset is a **view**, not a copy. *Which* boundary
selects that run depends on the mode, and the two must not be conflated:

| Mode | Selected by | Why |
| :--- | :--- | :--- |
| window | the requested **interval** | the interval *is* the request; the wider passband would return pairs nobody asked for |
| count | the filter's **passband**, then the `howmany` nearest the target | the interval is an internal DoS-sized artifact — trimming there would make the returned count depend on the stochastic KPM error |

The passband is the *trust* boundary: where the filter polynomial sits above cutoff, so Ritz
ordering faithfully tracks energy ordering. Outside it a Chebyshev tail can lift a distant
eigenvalue above a nearer one, which is why count mode gates there first — inside the passband,
"nearest the target" and "contiguous run inward" coincide.

In window mode the returned run is widened by one pair on each side, giving **brackets** that
certify the window edges were passed. A bracket lies just outside the passband by construction,
so it need not be the immediate spectral neighbour.

`PolfedResult` destructures as `(vals, vecs, report)`. There is deliberately **no** `length`
method — the count is `length(result.vals)` — and no `getindex`; use
[`eigenvector`](@ref UniPolfed.eigenvector) instead. If fewer than `howmany` pairs land inside
the interval it **warns rather than throwing**.

### [`PatchedResult`](@id UniPolfed.PatchedResult)

Returned by `patched_polfed` and `full_polfed`. Fields:

| Field | Contents |
| :--- | :--- |
| `vals` | eigenvalues, ascending, true energies of ``H`` |
| `vecs` | eigenvectors — a dense matrix in count/full mode, a **vector of vectors** in window mode |
| `report` | the aggregate [`PatchedReport`](@ref UniPolfed.PatchedReport) |
| `layout` | the Krylov block layout a matrix `vecs` uses |

There is no `residuals` field: a patched run has no single convergence history, so per-eigenpair
residuals live in the per-patch reports inside `report`.

### Per-eigenvector access

### [`eigenvector`](@id UniPolfed.eigenvector)

```julia
eigenvector(result, i) -> AbstractVector
```

The `i`-th eigenvector as a length-``D`` view, for either result type.

Returned eigenvectors are **always columns**: `vecs[:, i]` is the `i`-th eigenvector for every
entrypoint, regardless of the Krylov layout the run used internally. This accessor returns a view
rather than a copy, and additionally covers **window-mode** patched runs, where `vecs` is a vector
of independently owned vectors rather than a matrix.

### [`eigenpair`](@id UniPolfed.eigenpair)

```julia
eigenpair(result, i) -> (val, vec)
```

The `i`-th `(eigenvalue, eigenvector)` pair.

```julia
for i in eachindex(result.vals)
    λ, v = eigenpair(result, i)
end
```

## Report types

### [`Report`](@id UniPolfed.Report)

A standard report combines a [`MappingReport`](@ref UniPolfed.MappingReport), a
[`SpectralTransformReport`](@ref UniPolfed.SpectralTransformReport), the Lanczos
[`FactorizationReport`](@ref UniPolfed.Numerics.Lanczos.FactorizationReport), and stage
timings.

### [`MappingReport`](@id UniPolfed.MappingReport)

Snapshot of the resolved operator, rescaling, layout, processing unit and parallelization
strategy.

### [`SpectralTransformReport`](@id UniPolfed.SpectralTransformReport)

Snapshot of the target interval, selected filter, polynomial order, cutoff and estimated
transformation cost.

### [`FactorizationReport`](@id UniPolfed.Numerics.Lanczos.FactorizationReport)

Lanczos convergence history, residuals, Krylov-budget usage and internal timings.

### [`PatchReport`](@id UniPolfed.PatchReport)

Diagnostics for one interval solved by a patched or full run: kind, filter, order, interval,
budget, the complete per-run `FactorizationReport`, coverage and retry count.

A run's **kind** (`:edge_lowest`, `:edge_highest`, `:interior`) is not its filter **type** —
an interior patch near ``\pm 1`` can use an `EdgeFilter`, so the `EdgeFilter` count in a
histogram is always at least the number of edge peels.

### [`PatchedReport`](@id UniPolfed.PatchedReport)

Aggregate report for a patched or full run, carrying one `PatchReport` per run. Sections:

- **Decomposition** — run counts, filter histogram, order min/max/mean, aggregate cost
  ``\sum_p K_p \cdot bs_p \cdot iters_p``.
- **Mapping**.
- **Convergence** — aggregated; the quoted figure is the worst **eigen** residual, not the
  pool's worst Ritz residual.
- **Timings**.

## [Displaying a report](@id api_display_report)

### [`display_report`](@id UniPolfed.display_report)

```julia
display_report(r; io = stdout, convergence_details = false, color = auto)
display_report(io::IO, r; convergence_details = false, color = auto)
```

The print verb for every result and report type on this page — `PolfedResult`,
`PatchedResult`, `DoublePolfedResult`, `Report`, `PatchedReport`:

```julia
result = polfed(H, x0, howmany, target)
display_report(result)         # header followed by the complete report
display_report(result.report)  # report only
```

### What each entrypoint prints

All four outputs below come from the same disordered XXZ chain
(`xxz_hamiltonian(12, 6, 1.0, 0.7, 0.5)`, ``\mathcal{D} = 924``; the full-spectrum one uses
``L = 10``, ``\mathcal{D} = 252``). The **header** tells you which result type you have; the
**sections** below it come from the report.

`polfed` → a [`PolfedResult`](@ref UniPolfed.PolfedResult), four sections:

```text
POLFED result
════════════════════════════════════════════════════════════

Spectral Transformation ────────────────────────────────────────────────
- target: :maxdos
- interval: resc. [+2.75e-01,+2.94e-01], unresc. [+3.56e-01,+4.36e-01]
- filter: DeltaUpFilter · order K = 268 · cutoff 0.14
- mat-vec muls: 18_512
Mapping ────────────────────────────────────────────────────────────────
- operator: generic operator (mul! / user f!)
- layout: RowLayout    block size 1    opdim 924
- parallel strategy: BlockWise
- Energy: Emin = -5.006   Emax = 3.409
Convergence (block lanczos) ────────────────────────────────────────────
- Number of converged eigenpairs: 20 (20 requested)
- Ritz  conv tol=1.0e-14 passed:  20 (max res. 1.45e-41)
- Eigen conv tol=1.0e-09 passed:  20 (worst examined 2.78e-15)
- Iterations needed: 69 (out of 87 reserved, overestimated by 20.69%)
- Returned set: certified complete
Timings ────────────────────────────────────────────────────────────────
- Total polfed run took: 6.89 seconds (walltime), 6.85 seconds (CPU time)
- Walltime distribution: Lanczos 75.24% · preparation 24.76%
- Lan walltime: matvec 3.80% · reorthog. 34.87% · tridiag. 26.79% · conv.check 34.55%
- Lan CPU time: matvec 3.80% · reorthog. 34.87% · tridiag. 26.78% · conv.check 34.55%
```

`patched_polfed` → a [`PatchedResult`](@ref UniPolfed.PatchedResult). **Spectral
Transformation** is replaced by **Spectral Decomposition** (there are N filters, not one), and
Convergence aggregates across runs:

```text
Patched POLFED result
════════════════════════════════════════════════════════════
eigenpairs returned: 60

Spectral Decomposition ─────────────────────────────────────────────────
- Window: resc. [0.2527, 0.3152]  (true [0.2646, 0.5274])
- Runs: 3 — 3 interior patches (Window target) + 0 edge peel(s) (count target)
- Sizing: howmany_per_patch=20, howmany_edge=0, howmany_pad=3
- Filters: 3× DeltaUpFilter
- Polynomial order: min 175, max 198, mean 188.7 (max drives the cost)
- Cost: H·v = Σ Kₚ·bsₚ·itersₚ = 67690 mat-vec muls
Mapping ────────────────────────────────────────────────────────────────
- operator: generic operator (mul! / user f!)
- layout: RowLayout    block size 1    opdim 924
- parallel strategy: BlockWise
Convergence (across 3 runs) ────────────────────────────────────────────
- Eigenpairs kept (glued): 60 (78 converged across runs)
- Runs short of coverage: 0 / 3
- Worst eigen residual (‖H·v−λ·v‖) across runs: 1.86e-14
- Krylov iterations: 360 total (of 618 reserved)
```

`full_polfed` → the same `PatchedResult`, so the same shape. What differs is the window
(all of ``[-1,1]``), the edge peel, and the mixed filter histogram:

```text
Patched POLFED result
════════════════════════════════════════════════════════════
eigenpairs returned: 252

Spectral Decomposition ─────────────────────────────────────────────────
- Window: resc. [-1.0000, 1.0000]  (true [-4.155, 2.827])
- Runs: 9 — 8 interior patches (Window target) + 1 edge peel(s) (count target)
- Sizing: howmany_per_patch=30, howmany_edge=10, howmany_pad=3
- Filters: 8× DeltaUpFilter, 1× EdgeFilter
- Polynomial order: min 3, max 35, mean 21.2 (max drives the cost)
- Cost: H·v = Σ Kₚ·bsₚ·itersₚ = 28067 mat-vec muls
Convergence (across 9 runs) ────────────────────────────────────────────
- Eigenpairs kept (glued): 252 (297 converged across runs)
- Runs short of coverage: 0 / 9
- Worst eigen residual (‖H·v−λ·v‖) across runs: 3.32e-13
- Krylov iterations: 1366 total (of 2229 reserved)
```

Here coverage *is* completeness: `Runs short of coverage: 0 / 9` and 252 of
``\mathcal{D} = 252`` pairs mean the spectrum was fully resolved.

`double_polfed` → a `DoublePolfedResult`: a decision header, then **both** `PolfedResult`s
printed in the order you passed the targets:

```text
double POLFED result
════════════════════════════════════════════════════════════
  strategy : double  — one joint DoubleDeltaFilter run  (K_d = 310, from K₁ = 344, K₂ = 334)
  decision : ratio = 0.971  (threshold 0.400, reason :ratio)
  windows  : 15 + 15 eigenpairs
  note     : both windows come from ONE Lanczos run; the two reports below describe that same run.

── window 1 ─────────────────────────────────────────────────
POLFED result
════════════════════════════════════════════════════════════

Spectral Transformation ────────────────────────────────────────────────
- target: (:unrescaled, -0.3)
- interval: resc. [+1.11e-01,+1.26e-01], unresc. [-3.32e-01,-2.68e-01]
- filter: DoubleDeltaFilter · order K = 310 · cutoff 0.24
- mat-vec muls: 25_745
…

── window 2 ─────────────────────────────────────────────────
POLFED result
════════════════════════════════════════════════════════════

Spectral Transformation ────────────────────────────────────────────────
- target: (:unrescaled, 0.35)
- interval: resc. [+2.65e-01,+2.81e-01], unresc. [+3.18e-01,+3.82e-01]
- filter: DoubleDeltaFilter · order K = 310 · cutoff 0.24
- mat-vec muls: 25_745
…
```

On the **joint** path both windows come out of one Lanczos run, so the two reports carry
identical Convergence and Timings figures — they describe that shared run, not each window's
share of it. The `decision` line shows what the strategy was chosen on: `ratio = 0.971` is well
above the `0.400` threshold, so one joint run won.

Two keywords tune the rendering:

| Keyword | Default | Effect |
| :--- | :--- | :--- |
| `color` | auto | colored on a terminal, plain for every other sink |
| `convergence_details` | `false` | appends the per-check (or per-run) table |

`io` is accepted positionally as well, which is what you want for a file — `color` then
defaults to `false`, so nothing but text reaches the file:

```julia
open(io -> display_report(io, result.report; convergence_details = true),
     "polfed-report.txt", "w")
```

### The `show` methods underneath

`display_report` renders nothing of its own: it packs those two keywords into an `IOContext`
and calls `Base.show(::IO, ::MIME"text/plain", …)`, which is where every report type's
rendering lives. So `display(result)`, `show(io, MIME"text/plain"(), result)` and typing a
result at the REPL all print the same thing, and the two switches are equally available as
**IO properties** — `:color` and `:convergence_details` — for when they should travel with a
sink rather than be spelled at the call site:

```julia
show(IOContext(stdout, :convergence_details => true), MIME"text/plain"(), result.report)
```

`display_report` exists because `Base.display` accepts no keyword arguments —
`display(report; convergence_details = true)` is a `MethodError`.

## [Logging](@id api_logging)

Every solver keyword `verbosity` raises the level for **one call** and restores it afterwards;
`UniPolfed.Common.verbosity[]` sets it globally, and
[`with_verbosity`](@ref UniPolfed.Common.with_verbosity) scopes it around a block.

```julia
polfed(H, x0, 50, :maxdos; verbosity = 3)     # trace this run only
UniPolfed.Common.verbosity[] = 2              # …or set the level globally
```

| Level | Constant | Emits |
| :--- | :--- | :--- |
| `0` | [`POLFED_SILENT_LEVEL`](@ref UniPolfed.Common.POLFED_SILENT_LEVEL) | nothing |
| `1` | [`POLFED_WARN_LEVEL`](@ref UniPolfed.Common.POLFED_WARN_LEVEL) | potential issues, suboptimal setup |
| `2` | [`POLFED_INFO_LEVEL`](@ref UniPolfed.Common.POLFED_INFO_LEVEL) | one line per pipeline stage |
| `3` | [`POLFED_DEBUG_LEVEL`](@ref UniPolfed.Common.POLFED_DEBUG_LEVEL) | bound probes, filter ladder, per-check convergence |
| `4` | [`POLFED_DEEP_DEBUG_LEVEL`](@ref UniPolfed.Common.POLFED_DEEP_DEBUG_LEVEL) | per-iteration tracing |

Each level strictly adds to the one below. Level `0` is genuinely silent: POLFED routes its
own warnings through [`polfed_log`](@ref UniPolfed.Common.polfed_log) rather than `@warn`,
so `verbosity[] = 0` suppresses those too. Level `4` scales with the run and can reach
thousands of lines.

Everything a run decides happens *before* Lanczos starts, so that is where the trace is richest:

- **Level 2** reads as the run's story, one line per stage — mapping (`Emin`/`Emax`/`a`/`b`),
  DoS, the resolved target, the interval and how many states it holds, the resolved filter, then
  a **pre-Lanczos summary**: the filter and its order, the exposed window in *both* rescaled and
  true-energy coordinates, the `EigTarget` stop rule the run is given, the Krylov budget, and the
  projected `H·v` cost (`K · blocksize` per iteration — POLFED's bottleneck). The run closes with
  the converged count, iterations and residuals.
- **Level 3** adds the decisions behind those lines: the DoS sum-rule check and density profile,
  the max-DoS search, the window bisection, **every filter-ladder rung with the reason it was
  rejected**, the Clenshaw coefficient profile, the tolerances in the units they are applied in
  (a window run measures Ritz values against the *rescaled* operator), and the selection rule
  that turned the Lanczos output into the returned eigenpairs.
- **Level 4** adds the inner loops: order bisection, window-bisection steps, where a filter leaks,
  and per-iteration Lanczos tracing.

```@docs
UniPolfed.Common.POLFED_SILENT_LEVEL
UniPolfed.Common.POLFED_WARN_LEVEL
UniPolfed.Common.POLFED_INFO_LEVEL
UniPolfed.Common.POLFED_DEBUG_LEVEL
UniPolfed.Common.POLFED_DEEP_DEBUG_LEVEL
UniPolfed.Common.verbosity
UniPolfed.Common.with_verbosity
UniPolfed.Common.should_log
UniPolfed.Common.polfed_log
```

## Defaults

Defaults live in the registry `UniPolfed.Defaults`, as one submodule **per entrypoint** plus one
**cross-cutting** submodule for filters:

- `Defaults.polfed` — DoS, mapping and factorization defaults for the standard solver.
  `patched_polfed` also uses these for its public `mc`, `dosc`, `tol`, `eigentol` and
  `overestimate_iters` keywords.
- `Defaults.patched_polfed` — patched decomposition and its internal factorization defaults.
- `Defaults.full_polfed` — the full-spectrum equivalents.
- `Defaults.double_polfed` — the two-window knobs (notably `double_ratio`, `krylov_overhead`).
- `Defaults.filters` — filter shape defaults, one submodule per filter
  (`Defaults.filters.delta_up.cutoff`, `Defaults.filters.edge.order`, …), plus the shared
  geometry knob `Defaults.filters.edge_tol` and the resolution chains
  `INTERIOR_FILTER_CHAIN` / `EDGE_FILTER_CHAIN`. A *sibling* of the entrypoint modules, not a
  child of one, because every entrypoint resolves filters.

Override them through [`DoSConfig`](@ref UniPolfed.DoSConfig),
[`MappingConfig`](@ref UniPolfed.MappingConfig) or solver keywords rather than by modifying
the constants.

### Standard `polfed` defaults

- DoS: `N = 500`, `R = 500`, `kernel = :Jackson`.
- Mapping: `optimize_mapping = false`, `maxdos_edge_margin = 0.01`.
- Lanczos: `rot = FullRO()`, `basistype = MatrixBasis`, `tol = 1e-14`, `eigentol = 1e-9`,
  `overestimate_iters = 1.25`.
- Convergence checks start at `check_start_frac = 1.0`; the `conv_checking(expected_dim)`
  callback then adapts the interval between checks to the current residual.

### Patched and full defaults

Both registries use `howmany_pad = 3` and `max_glue_retries = 3`. `retry_budget_growth` is
`2.0` for `patched_polfed` and `1.5` for `full_polfed`.
