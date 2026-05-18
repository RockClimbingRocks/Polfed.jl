# Polfed.jl

```@meta
CurrentModule = Polfed
```

`Polfed.jl` implements Polynomial Filtering Exact Diagonalization (POLFED):
polynomial spectral transformation plus Lanczos-type factorization for
computing eigenpairs in selected spectral regions of quantum many-body
Hamiltonians.

## Quick Start

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

L_loc = 12
L_grain = 2
g0 = 1.0
α = 0.5

mat = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
x0 = rand(size(mat, 1))
x0 ./= norm(x0)

vals, vecs = polfed(mat, x0, 100, 0.0)
```

## Where To Go Next

- [Getting Started](getting-started/index.md)
- [Choosing Target](tutorials/beginner/choosing-target/index.md)
- [Core Functions](documentation/core-functions/index.md)
- [Hamiltonian Models](documentation/models/index.md)
