# Multi-window / double-filter TODO — single-window assumptions to fix

`run_polfed_pipeline` (and its downstream) is written around **one target window / one filter**.
A `DoubleDeltaFilter` (and any future multi-window filter) exposes **two** windows, so several
single-window assumptions must be lifted before it runs end-to-end. This file lists them.

## Already done

- `resolve_eig_target(exposed_end(filter), howmany)` in `src/main/logic/filter/filter.jl` maps
  `:highest → HighestEigs`, `:lowest → LowestEigs`, **`:both → BothEnds(howmany, howmany)`**
  (each end gets `howmany`). Wired into the pipeline. ✅
- `DoubleDeltaFilter` coefficients pin **both** peaks exactly (`P(target1)=+1`, `P(target2)=-1`)
  via a 2×2 solve. ✅

## Still single-window (must change for a double filter)

### 1. Filter selection only produces ONE window
- `src/main/pipeline/run_polfed_pipeline.jl` does: `resolve_target(...)` → `resolve_interval(...)`
  → **one** `Interval` → `resolve_filter(interval)` (single-interval orchestrator, picks only
  single-window filters). So the normal pipeline **never builds a `DoubleDeltaFilter`**.
- The multi-window entry `resolve_filter(::AbstractVector{Interval})` in
  `src/main/logic/filter/resolve_filter.jl` currently just `error(...)`s "not implemented yet".
- **Needed:** a path that resolves TWO windows (from the DoS / target) and dispatches to the
  double/multi-window filter, i.e. implement `resolve_filter(::Vector{Interval})` + a way for the
  pipeline to decide it wants two windows.

### 2. Result filtering keeps only ONE window  ← most likely to bite silently
- `build_polfed_result` in `src/main/logic/report/build_report.jl`:
  `_interval_energy_window(mp, interval)` maps the **one** `target_interval` to a true-energy
  window, then `searchsortedfirst/last` takes a **single contiguous slice** of the sorted vals.
- A `BothEnds` run returns eigenpairs at **both** windows (low *and* high). With one interval,
  this keeps one window and **silently drops the other end's eigenpairs** (they're not contiguous
  with the first window in the sorted spectrum).
- **Needed:** filter to the **union of both** true-energy windows (two `searchsorted` slices, or a
  per-window loop), and keep the eigenvector subset for both.

### 3. The report describes ONE window
- `SpectralTransformReport` in `src/main/logic/report/SpectralTransformReport.jl` carries a single
  `target` / `left` / `right` / energy window (one row in the "Spectral Transformation" section).
- **Needed:** represent two targets / two windows (or a list), and render both.

## Suggested order of work

1. Two-window `resolve_filter(::Vector{Interval})` + build the `DoubleDeltaFilter`.
2. Two-window result filtering in `build_polfed_result` (#2 — the silent-drop one).
3. Two-window `SpectralTransformReport` + its rendering.
4. (`resolve_eig_target(:both)` is already done.)

## Unrelated but blocking `DoubleDeltaFilter` build (WIP bugs, see review)

- `src/main/logic/filter/filters/double_delta/build.jl`: `resolve_order!` uses undefined `fiter`
  (typo for `filter`, lines ~31/35) → crashes when called.
- `analytical_order(::DoubleDeltaFilter)` references non-existent `filter.target` (dead + broken).
- `resolve_bounds_from_order!(::DoubleDeltaFilter)` still assumes single `order`/`target`/`interval`
  fields (being reworked separately).
