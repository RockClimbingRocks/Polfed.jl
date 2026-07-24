# [Reporting](@id tutorial_reporting)

Reporting helps you understand what happened during a run and where time was
spent.

## Get a Report

[`polfed`](@ref UniPolfed.polfed) always returns a
[`PolfedResult`](@ref UniPolfed.PolfedResult), so no reporting flag is needed.
Typing the result at the REPL displays the report; its structured data is also
available as `result.report`.

```julia
result = polfed(mat, x0, howmany, target)
display(result)

vals = result.vals
vecs = result.vecs
report = result.report
```

## Example Report

A concrete example is often the easiest way to read the report. For example,
with a small QSun test problem:

```julia
using UniPolfed: polfed
using UniPolfed.Models: qsun_hamiltonian
using LinearAlgebra
using Random

rng = MersenneTwister(1234)

L_loc = 6
L_grain = 2
g0 = 1.0
α = 0.5
mat = qsun_hamiltonian(L_loc, L_grain, g0, α; rng=rng, use_sparse=true)

x0 = rand(rng, size(mat, 1))
x0 ./= norm(x0)

howmany = 20
target = :middle

result = polfed(mat, x0, howmany, target)
display(result)
```

The exact values change with the Hamiltonian, random seed, target, and hardware,
but a standard report always has these sections:

```text
POLFED result
============================================================

Spectral Transformation ------------------------------------
Mapping ----------------------------------------------------
Convergence (lanczos) --------------------------------------
Timings ----------------------------------------------------
```

## What the Report Shows

- Spectral transformation summary:
  target strategy, rescaled and true-energy intervals, filter type, polynomial
  order, cutoff, and matrix multiplication count
  ([`SpectralTransformReport`](@ref UniPolfed.SpectralTransformReport)).
- Mapping summary:
  operator choice, Krylov layout, block size, Hilbert-space dimension,
  parallel strategy, and spectral bounds
  ([`MappingReport`](@ref UniPolfed.MappingReport)).
- Factorization summary:
  Lanczos/Block Lanczos type, block size, converged eigenpair counts, Lanczos
  and eigen residuals, completeness, and reserved versus used iterations
  ([`FactorizationReport`](@ref
  UniPolfed.Numerics.Lanczos.FactorizationReport)).
- Timing summary:
  wall/CPU time for the run, a Lanczos-versus-preparation split, and Lanczos
  time divided among matrix-vector products, reorthogonalization,
  tridiagonalization, and convergence checks.
- Optional convergence table:
  per-check evolution of Krylov dimension, converged count, and residual.

The difference between standard Lanczos and Block Lanczos is explained in
[Lanczos and Block Lanczos Factorization](@ref tutorial_lanczos_block).

## Display Convergence Details

```julia
detail_io = IOContext(
    stdout,
    :color => true,
    :convergence_details => true,
)
show(detail_io, MIME"text/plain"(), result.report)
```

The table is hidden by default. Enable it when studying convergence or when a
run exhausts its Krylov budget. To write a report without ANSI color escapes,
use an `IOContext` with `:color => false`; the complete file example is in
[Reports, Logging, and Defaults](@ref docs_report_defaults).

## Patched Reports

`patched_polfed` and `full_polfed` return a
[`PatchedReport`](@ref UniPolfed.PatchedReport):

```julia
using UniPolfed: patched_polfed

vals, vecs, report = patched_polfed(
    mat,
    x0,
    howmany,
    howmany_per_patch,
    target,
)
display(report)
```

Its sections summarize the spectral decomposition, mapping, convergence across
all patch runs, and aggregate timings. The same `:convergence_details => true`
IO property adds a per-run table.

## Logging Levels

UniPolfed.jl also offers different logging levels to better understand what is
going on and if and when does the problem occur.

```julia
using UniPolfed

UniPolfed.Common.verbosity[] = 0 # silent
UniPolfed.Common.verbosity[] = 1 # warn
UniPolfed.Common.verbosity[] = 2 # info
UniPolfed.Common.verbosity[] = 3 # debug
UniPolfed.Common.verbosity[] = 4 # deep debug
```

These levels represent the following:

- `0` ([`POLFED_SILENT_LEVEL`](@ref UniPolfed.Common.POLFED_SILENT_LEVEL)):
  no logging.
- `1` ([`POLFED_WARN_LEVEL`](@ref UniPolfed.Common.POLFED_WARN_LEVEL)):
  WARN messages only (potential issues / suboptimal setup).
- `2` ([`POLFED_INFO_LEVEL`](@ref UniPolfed.Common.POLFED_INFO_LEVEL)):
  INFO + WARN (major run phases and setup diagnostics) — one line per
  pipeline stage: mapping, DoS, target interval, filter, transform, Lanczos.
- `3` ([`POLFED_DEBUG_LEVEL`](@ref UniPolfed.Common.POLFED_DEBUG_LEVEL)):
  DEBUG + INFO + WARN (detailed tracing: bound probes, filter-ladder
  attempts, per-check convergence events).
- `4` ([`POLFED_DEEP_DEBUG_LEVEL`](@ref UniPolfed.Common.POLFED_DEEP_DEBUG_LEVEL)):
  DEEP + DEBUG + INFO + WARN. Adds **per-iteration** tracing, so its volume
  scales with the run (hundreds to thousands of lines). Use it to pin down
  *where inside a stage* something goes wrong; too noisy for production runs.

Each level strictly adds to the one below it, and level `0` is genuinely
silent — POLFED's own warnings are routed through
[`polfed_log`](@ref UniPolfed.Common.polfed_log), so `verbosity[] = 0`
suppresses them too.

For full report/logging reference, see
[Reports, Logging, and Defaults](@ref docs_report_defaults).

Use higher levels when diagnosing convergence/setup issues.
