# Getting Started

This page takes you from installation to a minimal working POLFED workflow.

## Installation

Install from the Julia package manager:

```julia
import Pkg
Pkg.add("UniPolfed")
```

Then load the package in your session:

```julia
using UniPolfed
```

## Minimal Working Example

The run below illustrates the full minimal pipeline:
1. Build a matrix (here a real, symmetric many-body Quantum Sun Hamiltonian;
   see [Quantum Sun (QSun)](@ref qsun_model)),
2. Either construct one normalized initial state (vector input) or a
   normalized block of states (matrix input),
3. Call [`polfed`](@ref UniPolfed.polfed).

```julia
using UniPolfed
using UniPolfed.Models: qsun_hamiltonian
using LinearAlgebra

# Problem setup
L_loc = 12
L_grain = 2
g0 = 1.0
α = 0.5
mat = qsun_hamiltonian(L_loc, L_grain, g0, α; use_sparse=true)
howmany = 100 # howmany eigenpairs to target
target = 0.0 # What part of the spectrum to target

# Vector input -> Lanczos factorization
x0_vec = rand(size(mat, 1)); x0_vec ./= norm(x0_vec)
result_l = polfed(mat, x0_vec, howmany, target)

# Matrix input -> Block Lanczos factorization
x0_mat = rand(size(mat, 1), 4)
x0_mat = Matrix(qr(x0_mat).Q)
result_b = polfed(mat, x0_mat, howmany, target)

# Eigenvalues, eigenvectors, and diagnostics are fields of each PolfedResult
vals_l = result_l.vals
vecs_l = result_l.vecs
```

Interpretation:
- `x0_vec` requests standard Lanczos.
- `x0_mat` requests Block Lanczos with block size equal to column count.
- Both entry modes are handled by the same [`polfed`](@ref UniPolfed.polfed)
  interface.

For a more detailed comparison, see
[Lanczos and Block Lanczos Factorization](@ref tutorial_lanczos_block).

## First Diagnostics

Reporting is the fastest way to understand runtime behavior:

```julia
result = polfed(mat, x0_vec, howmany, target)
display(result)
```

The report shows polynomial order, matrix multiplications, convergence,
factorization details, and timing split. See
[`Report`](@ref UniPolfed.Report); it is always available as `result.report`.
*
