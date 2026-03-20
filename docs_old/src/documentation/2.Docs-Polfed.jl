# # Polfed API Reference
#
# This page documents the main `polfed` entry points and the configuration structs
# used to control the algorithm.
#
# ```@docs
# Polfed.polfed
# ```
#
# ## Minimal Example
#
# ```julia
# using Polfed
# using Polfed.QSun: quantum_sun_hamiltonian
# using LinearAlgebra
#
# mat = quantum_sun_hamiltonian(12, 2; sparse=true)
# v0 = rand(size(mat, 1)); v0 ./= norm(v0)
# howmany = 100
# target = :maxdos
#
# vals, vecs = polfed(mat, v0, howmany, target)
# ```
#
# ## Target Specification
#
# The `target` argument can be numeric (explicit energy) or symbolic:
# - `:maxdos` (default interpretation of previous `nothing`)
# - `:middle`
# - `(:quantile, q)` where `q ∈ [0,1]`
# - `(:offset, frac)` offset from the DoS peak toward `Emax`
# - `(:absolute, E)` explicit energy
#
# ```@docs
# Polfed.TargetSpec
# Polfed.TargetMaxDoS
# Polfed.TargetMiddle
# Polfed.TargetQuantile
# Polfed.TargetOffset
# Polfed.TargetAbsolute
# Polfed.normalize_target_spec
# ```
#
# ## Mapping Configuration
#
# Mapping config controls:
# - parallelization strategy
# - optimized mapping and rescaling (Emin/Emax)
# - optional custom rescaled mapping and/or Clenshaw kernels
#
# ```@docs
# Polfed.MappingConfig
# ```
#
# Example:
# ```julia
# mapping = MappingConfig(
#     parallelization = MulColsParallel(),
#     optimize_mapping = true,
# )
#
# vals, vecs = polfed(mat, v0, howmany, :maxdos; mapping=mapping)
# ```
#
# ## Transform Configuration
#
# Transform config controls polynomial behavior:
# - coefficients, normalization, cutoff
# - target interval (`left`, `right`)
# - polynomial order and safety factor
#
# ```@docs
# Polfed.TransformConfig
# ```
#
# Example (custom coefficients):
# ```julia
# transform = TransformConfig(
#     coefficients = (λ, n) -> exp(-n/50) * cos(n * acos(λ)),
#     cutoff = 0.2,
# )
#
# vals, vecs = polfed(mat, v0, howmany, :middle; transform=transform)
# ```
#
# ## Factorization Configuration
# ```@docs
# Polfed.FactorizationConfig
# ```
#
# ## Density of States Configuration
# ```@docs
# Polfed.DoSConfig
# ```
#
# ## Complex Matrices and GPU Notes
#
# - Complex Hermitian matrices are supported on CPU (eigenvalues are real).
# - Complex GPU arrays are not supported yet; use CPU arrays for complex problems.
