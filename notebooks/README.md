# UniPolfed notebooks

This is the curated, source-backed notebook set. It keeps the reproducible experiments from the
older `Polfed/notebooks` tree without manifests, Conda environments, generated plots, datasets, or
machine-specific runner scripts.

## Setup and execution

From the repository root:

```bash
julia --project=notebooks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

The `[sources]` entry in `Project.toml` points `UniPolfed` at the parent checkout. The explicit
`Pkg.develop` call keeps setup compatible with Julia versions that do not yet honor project
sources.

Open the `.ipynb` files in a Julia-capable notebook editor such as VS Code, or launch
`jupyter lab --notebook-dir=notebooks` when a working Jupyter installation is available. Select
the `notebooks` Julia environment for the kernel.

Open the notebooks with the `notebooks` environment:

- `qrem_roundtrip.ipynb` checks count, regular-window, patched-window, and full-spectrum entrypoints
  against exact diagonalization. The regular window intentionally returns an eigenpair bracketing
  each finite window edge; comparisons therefore filter to the requested window first.
- `filters/double_delta.ipynb` builds and validates the production double-delta filter.
- `filters/order_analysis.ipynb` compares the analytical order estimate with resolved orders.
- `benchmarks/lanczos_convergence.ipynb` records a small resumable convergence smoke benchmark.

All committed notebooks have empty output cells. Before committing an executed notebook, strip it:

```bash
jupyter nbconvert --clear-output --inplace notebooks/qrem_roundtrip.ipynb
jupyter nbconvert --clear-output --inplace notebooks/filters/double_delta.ipynb
jupyter nbconvert --clear-output --inplace notebooks/filters/order_analysis.ipynb
jupyter nbconvert --clear-output --inplace notebooks/benchmarks/lanczos_convergence.ipynb
```

The defaults are smoke-sized (`L=8`) and are correctness/development checks, not publishable
benchmarks. The Lanczos notebook has a `FULL_SWEEP` switch for the former `L=16` experiment; expect
substantially greater time and memory use. Its generated CSV and figures go under
`notebooks/results/`, which is ignored.

## Findings retained from the larger archive

- Window-mode `polfed` certifies completeness by returning the in-window eigenpairs plus one
  converged bracket on each finite side. Patched-window results own only the requested interval, so
  compare regular and patched runs after filtering the regular result to `[lo, hi]`.
- The analytical delta-filter order is a useful cutoff-independent search bracket. The production
  order resolver still has to enforce the selected cutoff, interval coverage, and leakage contract,
  so the final order can change with cutoff and target location.
- Double-delta filtering combines independently resolved `+1` and `-1` peaks. Both requested
  intervals must remain inside their resolved passbands; rebuilding through `build_filter` is more
  reliable than preserving hand-written coefficients.
- Mapping speedups in the excluded notebooks were tied to CPU model, thread count, matrix layout,
  kernel choice, and sometimes GPU hardware. They should be re-benchmarked on the target machine;
  archived timings are not portable evidence.
- The projected/banded diagonalization experiments were exploratory. Their simulations suggested a
  possible reduced-memory direction, but the evidence and correctness coverage were incomplete, so
  no banded dependency or production recommendation is carried into this project.

## Material intentionally left in the archive

The copy omits non-running optimization previews, notebooks tied to obsolete QSun APIs,
duplicate or plot-only explorations, raw machine-specific benchmarks, and the incomplete
projected-diagonalization prototype. Their reusable conclusions are condensed above and in the
[development roadmap](../docs/src/development/roadmap.md).
