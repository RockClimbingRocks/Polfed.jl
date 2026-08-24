# [FAQ / Troubleshooting](@id faq)

Common symptoms and where to look. For what the report *sections* mean, see the
"Reading the output" section of the tutorial you are following —
[POLFED](@ref tutorial_polfed), [Patched](@ref tutorial_patched),
[Full](@ref tutorial_full).

## Convergence

### Why did I get fewer than `howmany` eigenpairs?

POLFED warns rather than throwing when fewer pairs land inside the target interval than you
asked for. The usual causes, in order of likelihood:

1. **The Krylov budget ran out.** Check *Convergence → reserved versus used iterations* in
   the report. Raise `overestimate_iters` (say to `1.5`), or raise `howmany`.
2. **`eigentol` is unreachable.** In `Float32`/`ComplexF32`, `1e-9` cannot be met. Loosen it
   to match your element type.
3. **The filter admitted a wider passband than you requested.** Pairs outside your interval
   are trimmed at the end, so a wide passband converges pairs you then discard. That is
   working as intended.
4. **Degeneracies.** Exactly degenerate eigenvalues cannot be separated — see below.

### `converged = 0` and nothing obviously wrong

If your test Hamiltonian has a **symmetric spectrum** and you targeted the **center**, you
have hit exact degeneracy: Lanczos cannot split exactly degenerate eigenvalues. Use an
asymmetric Hamiltonian or an off-center target. This is the single most common false alarm
in a first test.

### The run exhausts its budget just short of converging

Raise `overestimate_iters` before anything else — it is cheaper than re-running with a
larger `howmany`, and it does not change the problem.

### A patched run reports a coverage warning

A patch failed to bracket the interval it owns even after its retries, so the returned set
has a **gap** — the result is incomplete, not wrong. The report says which interval. Try a
smaller `howmany_per_patch`, a larger `howmany_pad`, or a looser `eigentol`.

## Performance

### My run is slow. What do I check first?

1. `display_report(result)` and read *Spectral Transformation → polynomial order*. If ``K`` is
   large, your window is narrow — raising `howmany` widens it and lowers ``K``. See
   [The bottleneck](@ref going_faster).
2. *Timings → Lanczos versus preparation*. Preparation should be a small fraction.
3. *Timings inside Lanczos → matvecs versus reorthogonalization versus the projected
   diagonalization*. Matvecs dominating points at [kernels](@ref optimized_kernels) and
   [parallelization](@ref parallelization). Reorthogonalization or the dense diagonalization
   dominating means the Krylov space is too big — that is the regime
   [Patched POLFED](@ref tutorial_patched) is for, and lowering `block_size` may help too.

### I requested more eigenpairs and it got *faster*. Is that real?

Yes. A larger `howmany` widens the target interval, which **lowers** the polynomial order
``K``, and ``K`` multiplies every Lanczos matvec. It does not scale indefinitely: the Krylov
basis and the projected diagonalization both grow with `howmany`. See
[Guidelines](@ref guidelines).

### Threads do not seem to help

[`VectorWise`](@ref UniPolfed.VectorWise) splits the Krylov block by **vector**. With a
vector `x0` there is one vector and nothing to split. Either pass a matrix `x0` (block
Lanczos) or use [`BlockWise`](@ref UniPolfed.BlockWise), which parallelizes inside the kernel
and is the only strategy that helps a single vector.

### `VectorWise` leaves threads idle

It opens one task per Krylov vector, so a block with fewer vectors than threads cannot use
them all — POLFED warns when it sees this. Either match the block size to `nthreads()`, or use
[`BlockWise`](@ref UniPolfed.BlockWise), which splits the basis-state range instead. See
[Choosing a strategy](@ref parallelization_reasoning).

## Setup and configuration

### Which mapping keyword do I need?

`parallel_strategy` and `optimize_mapping`, both inside
[`MappingConfig`](@ref UniPolfed.MappingConfig). Start with `optimize_mapping = true` if
your matrix is structured.

### I passed `f!_rescaled` and nothing changed

`f!_rescaled` requires `Emin` **and** `Emax` in the same `MappingConfig` — POLFED needs the
``a`` and ``b`` you rescaled with in order to map results back. Without them it falls back
to rescaling at runtime. Check *Mapping → operator* in the report.

### Lanczos or block Lanczos — how do I choose?

You do not pass a flag; the shape of `x0` decides. A vector selects Lanczos, a matrix
selects block Lanczos with block size equal to the column count. Keep
``\texttt{howmany}/\texttt{block\_size} \gtrsim 100``.

### `(:offset, 0.5)` and `(:rescaled, 0.5)` gave different answers

They are different by construction. `:rescaled` measures from the **center of the
spectrum**; `:offset` measures from the **DoS maximum** toward an edge. See
[Target modes](@ref polfed_targets).

### `:maxdos` moved between runs

`:maxdos` is the peak of a *stochastic* KPM estimate, so `N` and `R` in
[`DoSConfig`](@ref UniPolfed.DoSConfig) can shift it. Raise `R` for a more stable peak, `N`
for finer resolution.

## Errors

### "No valid filter" / the filter chain was exhausted

Every candidate cutoff failed validation, meaning POLFED could not build a filter for your
interval that is guaranteed not to leak. It raises rather than returning a filter that might
contaminate your results. Usually the requested window is too narrow relative to the
spectrum — widen it, or raise `howmany`.

### A `BoundsError` in threaded code

If it is in your own code: never index per-thread buffers by `Threads.threadid()`. Since
Julia 1.12 tasks migrate and `threadid()` is not bounded by the thread count. Use
chunk-based parallelism — split the work, give each task its own accumulator, reduce the
partials.

### `GPU()` throws on construction

The CUDA extension is unavailable or nonfunctional. Make sure `CUDA` is installed **and
loaded** (`using CUDA`) — the extension stays dormant until it is.

## Diagnosing anything else

```julia
UniPolfed.Common.verbosity[] = 3   # bound probes, filter ladder, per-check convergence
```

Level `3` is usually the right level for a setup problem; level `4` adds per-iteration
tracing and is for pinning down *where inside a stage* something goes wrong. Full table:
[Logging](@ref api_logging).
