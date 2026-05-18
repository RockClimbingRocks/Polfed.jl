# Polfed.jl

```@meta
CurrentModule = Polfed
```

`Polfed.jl` is a Julia package for polynomial filtering eigensolvers and
Hamiltonian tools for quantum many-body simulations.

```@raw html
<style>
.polfed-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  margin: 1.25rem 0;
}
.polfed-split-button {
  --polfed-segment-border: #dbdbdb;
  display: inline-flex;
  align-items: stretch;
  border: 1px solid var(--polfed-segment-border);
  border-radius: 8px;
  overflow: hidden;
}
.polfed-split-button .button {
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  margin: 0 !important;
}
.polfed-split-button .button:first-child {
  pointer-events: none;
}
.polfed-split-button .button:not(:first-child) {
  border-left: 1px solid var(--polfed-segment-border) !important;
}
html.theme--documenter-dark .polfed-split-button {
  --polfed-segment-border: #5e6d6f;
}
html.theme--catppuccin-latte .polfed-split-button {
  --polfed-segment-border: #acb0be;
}
html.theme--catppuccin-frappe .polfed-split-button {
  --polfed-segment-border: #626880;
}
html.theme--catppuccin-macchiato .polfed-split-button {
  --polfed-segment-border: #5b6078;
}
html.theme--catppuccin-mocha .polfed-split-button {
  --polfed-segment-border: #585b70;
}
.polfed-spectral-figure {
  margin: 1.75rem 0;
}
.polfed-spectral-figure img {
  display: block;
  width: 100%;
  max-width: 900px;
}
.polfed-spectral-dark {
  display: none !important;
}
html.theme--documenter-dark .polfed-spectral-light,
html.theme--catppuccin-frappe .polfed-spectral-light,
html.theme--catppuccin-macchiato .polfed-spectral-light,
html.theme--catppuccin-mocha .polfed-spectral-light {
  display: none !important;
}
html.theme--documenter-dark .polfed-spectral-dark,
html.theme--catppuccin-frappe .polfed-spectral-dark,
html.theme--catppuccin-macchiato .polfed-spectral-dark,
html.theme--catppuccin-mocha .polfed-spectral-dark {
  display: block !important;
}
</style>
<p class="polfed-actions">
  <a class="button is-primary" href="./">Documentation</a>
  <span class="polfed-split-button">
    <span class="button is-static">Citation</span>
    <a class="button is-primary" href="citation/#article">article</a>
    <a class="button is-primary" href="citation/#code">code</a>
  </span>
  <span class="polfed-split-button">
    <span class="button is-static">Code</span>
    <a class="button is-primary" href="https://github.com/RockClimbingRocks/Polfed.jl" target="_blank" rel="noopener noreferrer">GitHub</a>
  </span>
  <span class="polfed-split-button">
    <span class="button is-static">Article</span>
    <a class="button is-primary" href="https://scipost.org/SciPostPhysCodeb" target="_blank" rel="noopener noreferrer">SciPost</a>
    <a class="button is-primary" href="https://arxiv.org/abs/2605.10191" target="_blank" rel="noopener noreferrer">arXiv</a>
  </span>
</p>
```

If `Polfed.jl` supports your research, please cite both the overview article
and the code. This helps make the method and the software visible, reusable,
and easier to maintain for the community. Citation details are collected on the
[citation page](citation/index.md).

Version of the code: `v0.1.0`

```@raw html
<p class="polfed-actions">
  <a class="button is-primary" href="https://juliapkgstats.com/pkg/Polfed?timeframe=30d&trendingPeriod=14d&userData=true&ciData=true&missingData=true" target="_blank" rel="noopener noreferrer">Download Statistics</a>
</p>
```

## What Is POLFED?

Polynomial Filtering Exact Diagonalization (POLFED) is designed for eigenvalue
problems of the form

```math
H |\psi\rangle = E |\psi\rangle
```

where only a selected part of the spectrum is desired. The interface allows one
to target arbitrary spectral regions, while the method builds a polynomial
filter that amplifies components of a vector near the target energy $\lambda$ and
suppresses the rest of the spectrum. Krylov/Lanczos-type factorization is then
applied to the filtered problem.

```@raw html
<figure class="polfed-spectral-figure">
  <img class="polfed-spectral-light" src="assets/spectral-transform-webpage-light.svg" alt="Spectral transformation used by POLFED.">
  <img class="polfed-spectral-dark" src="assets/spectral-transform-webpage-dark.svg" alt="Spectral transformation used by POLFED.">
</figure>
```

The original POLFED method was introduced by Piotr Sierant and collaborators in
[PRL](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.156601)
/ [arXiv](https://arxiv.org/pdf/2005.09534).

For a broader overview and practical discussion, read the `Polfed.jl` article:
[SciPost](https://scipost.org/SciPostPhysCodeb)
/ [arXiv](https://arxiv.org/abs/2605.10191).

## Features and Capabilities

- Lanczos and block Lanczos factorizations are supported; see
  [Lanczos and Block Lanczos Factorization](tutorials/beginner/lanczos-block-lanczos/index.md).
- Arbitrary parts of the spectrum can be targeted; see
  [Choosing Target](tutorials/beginner/choosing-target/index.md).
- Built-in parallelization and optimized mapping workflows are available; see
  [Parallelization](tutorials/beginner/parallelization/index.md),
  [Optimized Mapping](tutorials/beginner/optimized-mapping/index.md), and
  [Reducing Memory Access](tutorials/beginner/reducing-memory-access/index.md).
- Automatic optimization and custom mappings can be used for structured models;
  see [Automatic Optimization](tutorials/advanced/automatic-optimization/index.md)
  and [Custom Mapping](tutorials/advanced/custom-mapping/index.md).
- CUDA workflows are supported where available; see
  [Working with GPUs](tutorials/beginner/working-with-gpus/index.md).
- Built-in Hamiltonian constructors are provided for
  [Quantum Sun (QSun)](models/qsun/index.md), [XXZ](models/xxz/index.md), and
  [J1-J2](models/j1j2/index.md).
- Reports expose convergence, timing, and configuration details; see
  [Reporting](tutorials/beginner/reporting/index.md) and
  [Reports, Logging, and Defaults](documentation/reports-logging-and-defaults/index.md).
