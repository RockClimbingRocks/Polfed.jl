# Notebooks

Research scripts and notebooks that use the package. They are not part of the
package and are not run by the test suite.

## Layout

- `rok/`, `Konrad/`, `Leonhard/`: one folder per person.
- `rok/Project.toml`: the notebook environment. It loads this checkout of `Polfed`
  through `[sources] Polfed = {path = "../.."}`, so notebooks always run against the
  working tree. Activate it with `--project=@.` (the default of the Jupyter kernels)
  or `--project=notebooks/rok`.
- `rok/test/parallel/`: prototypes for the Clenshaw and kernel work that was later
  moved into `src/`.
- `rok/test/private/`: work in progress. It is ignored by git, so it stays on the
  machine where it was written.

## What is not tracked

Generated output (`*.csv`, `*.h5`, `*.jld2`, `*.pdf`, `*.png`, `*.log`, `results/`),
the notebook `Manifest.toml`, and the `.CondaPkg/` Python environment are ignored.
Raw benchmark timings are ignored on purpose: they depend on the CPU, thread count, and
layout, so they are not portable evidence. Drivers, READMEs, and short written
summaries can be committed.
