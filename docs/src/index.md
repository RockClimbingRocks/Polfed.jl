# Polfed.jl Overview

```@meta
CurrentModule = Polfed
```

`Polfed.jl` implements Polynomial Filtering Exact Diagonalization (POLFED):
polynomial spectral transformation + Lanczos-type factorization to extract
eigenpairs around user-chosen spectral regions.

The documentation section is the canonical reference for solver entry points.
POLFED solves

```math
H |\psi\rangle = E |\psi\rangle
```

using polynomial filtering and Krylov factorization.

## Features and Capabilities

- Use both factorization modes through `x0` shape:
  - `x0::AbstractVector` -> **Lanczos Factorization**
  - `x0::AbstractMatrix` -> **Block Lanczos Factorization**
- Run real and complex workflows (`Float32`, `Float64`, `ComplexF32`, `ComplexF64`)
  with the same [`polfed`](@ref Polfed.polfed) interface, including CUDA-array workflows.
- Target arbitrary spectral regions with flexible target conditions:
  `:maxdos`, `:middle`, `(:offset, frac)`, `(:unrescaled, E)`,
  `(:rescaled, e)`, or plain numeric target. See
  [Choosing Target](@ref tutorial_choosing_target).
- Use built-in mapping parallelization with
  [`MulColsParallel`](@ref Polfed.MulColsParallel),
  [`TwoLevelParallel`](@ref Polfed.TwoLevelParallel), or
  [`NoParallel`](@ref Polfed.NoParallel) via
  [`MappingConfig`](@ref Polfed.MappingConfig) (`parallel_strategy=...`).
- Enable automatic mapping optimization with
  [`MappingConfig`](@ref Polfed.MappingConfig) (`optimize_mapping=true`).
  For Hamiltonians with only a few different offdiagonal values, [`polfed`](@ref Polfed.polfed) can
  separate diagonal/offdiagonal parts and significantly reduce memory traffic.
  See [Optimized Mapping](@ref tutorial_optimized_mapping) and
  [Reducing Memory Access](@ref Reducing_Memory_Access).
- Provide your own optimized mapping `f!(Y, X)` for model-specific speedups.
- Hamiltonian builders live under `Polfed.Models`; see
  [Quantum Sun (QSun)](@ref qsun_model), [XXZ](@ref xxz_model), and
  [J1-J2](@ref j1j2_model).
- Inspect performance/convergence diagnostics with
  [`produce_report`](@ref Polfed.PolfedCore.PolfedDefaults.produce_report),
  [`display_report`](@ref Polfed.display_report), and
  [Reports, Logging, and Defaults](@ref docs_report_defaults).

## Simplest Usage Example

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

L_loc = 12
L_grain = 2
g0 = 1.0
α = 0.5

mat = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
x0 = rand(size(mat, 1)); x0 ./= norm(x0)
howmany = 100
target = 0.0

vals, vecs = polfed(mat, x0, howmany, target)
```

## Citation

If you use `Polfed.jl` in your work, please cite:

```bibtex
@misc{Pintar26polfed,
  title         = {Computing eigenpairs of quantum many-body systems with Polfed.jl},
  author        = {Rok Pintar and Konrad Pawlik and Rafał Świętek and Miroslav Hopjan and Jan Šuntajs and Jakub Zakrzewski and Piotr Sierant and Lev Vidmar},
  year          = {2026},
  eprint        = {2605.10191},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.stat-mech},
  url           = {https://arxiv.org/abs/2605.10191},
}
```
