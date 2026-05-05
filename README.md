# Polfed.jl

[![Build Status](https://github.com/RockClimbingRocks/Polfed.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RockClimbingRocks/Polfed.jl/actions)
[![Documentation – stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RockClimbingRocks.github.io/Polfed.jl/stable)
[![Documentation – dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RockClimbingRocks.github.io/Polfed.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

**Polfed.jl** is a Julia package for performing **Polynomial Filtering (PolFed)** simulations of quantum systems.  
It provides efficient numerical routines for working with large Hamiltonians, including support for eigenvalue filtering, kernel polynomial methods, and custom iterative algorithms.

Polfed is designed for **high-performance simulations**, integrates smoothly with other Julia scientific computing packages, and includes GPU acceleration support (real-valued arrays). CPU runs support complex Hermitian matrices.

---

## Installation

For now, you can install directly from GitHub:

```julia
] add https://github.com/RockClimbingRocks/Polfed.jl
```

## Quick Start

```julia
using Polfed
using Polfed.QSun: qsun_hamiltonian
using LinearAlgebra

L_loc = 12
L_grain = 2
g0 = 1.0
α = 0.5

mat = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
v0 = rand(size(mat, 1)); v0 ./= norm(v0)
howmany = 100
target = :maxdos

vals, vecs = polfed(mat, v0, howmany, target)
```

### Config Split (Mapping vs Transform)

Polfed separates **mapping** concerns from **polynomial transform** concerns:

- `MappingConfig`: mapping kernel, rescaling (`Emin/Emax`), parallelization, and optimized mapping/clenshaw kernels.
- `TransformConfig`: coefficients, normalization, cutoff, interval (`left/right`), order, and safety factor.

Example:

```julia
mapping = MappingConfig(optimize_mapping=true)
transform = TransformConfig(cutoff=0.2)

vals, vecs = polfed(mat, v0, howmany, :middle; mapping=mapping, transform=transform)
```
