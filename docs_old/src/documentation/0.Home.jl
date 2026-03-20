# # Polfed.jl
#
# ```@meta
# CurrentModule = Polfed
# ```
#
# **Polfed.jl** is a Julia package for **Polynomial Filtering Exact Diagonalization (POLFED)**.
# It combines polynomial filtering with Lanczos / Block Lanczos factorization to extract
# eigenvalues and eigenvectors near a targeted energy region.
#
# Key ideas:
# - Build a polynomial filter `P(H)` that amplifies eigenvectors near a target.
# - Apply Lanczos / Block Lanczos to the filtered operator.
# - Use mapping kernels and Clenshaw recurrence to make the filter fast.
#
# ## Features
# - Fast polynomial filtering with Clenshaw recurrence
# - Standard and Block Lanczos factorization
# - Mapping optimizations for sparse Hamiltonians with structured off-diagonals
# - CPU multithreading + two-level parallelization
# - GPU support via CUDA.jl (real-valued arrays)
# - CPU support for complex Hermitian matrices (eigenvalues real)
#
# ## Installation
#
# ```julia
# import Pkg
# Pkg.add("Polfed")
# ```
#
# ## Quick Start
#
# ```julia
# using Polfed
# using Polfed.QSun: quantum_sun_hamiltonian
# using LinearAlgebra
#
# mat = quantum_sun_hamiltonian(12, 2; sparse=true)
# v0 = rand(size(mat, 1)); v0 ./= norm(v0)
# howmany = 100
#
# # Targeting examples:
# target = :maxdos           # peak of density of states (previous default)
# # target = :middle         # center of spectrum
# # target = (:quantile, 0.25)
# # target = 0.0             # explicit energy value
#
# vals, vecs = polfed(mat, v0, howmany, target)
# ```
#
# ## Config Split: Mapping vs Transform
#
# POLFED now separates configuration into two orthogonal parts:
# - `MappingConfig`: mapping kernel, rescaling (Emin/Emax), parallelization,
#   optimized mapping and optional custom Clenshaw kernels.
# - `TransformConfig`: polynomial coefficients, normalization, cutoff, target interval,
#   order and safety factor.
#
# This keeps the algorithm clean: mapping decisions are isolated from polynomial
# / spectral-transform decisions.
#
# ## GPU and Complex Notes
#
# - GPU execution works for **real** arrays.
# - Complex Hermitian matrices are supported on CPU (eigenvalues are real),
#   but **complex GPU arrays are not supported yet**.
#
# ## Documentation Flow
# - **Tutorials**: practical workflows and optimization recipes.
# - **API Docs**: full reference for configs, reports, and core functions.
