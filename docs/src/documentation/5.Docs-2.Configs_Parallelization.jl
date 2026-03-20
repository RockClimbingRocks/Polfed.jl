# # [Configs and Parallelization Types](@id docs_parallel_types)
#
# This page documents the main configuration structs and execution strategy
# types used by [`polfed`](@ref Polfed.polfed).
#
# ## [`MappingConfig`](@ref Polfed.MappingConfig)
#
# ```@docs
# Polfed.MappingConfig
# ```
#
# ## [`TransformConfig`](@ref Polfed.TransformConfig)
#
# ```@docs
# Polfed.TransformConfig
# ```
#
# ## [`FactorizationConfig`](@ref Polfed.FactorizationConfig)
#
# ```@docs
# Polfed.FactorizationConfig
# ```
#
# ## [`DoSConfig`](@ref Polfed.DoSConfig)
#
# ```@docs
# Polfed.DoSConfig
# ```
#
# ## Parallelization Strategy Types
#
# ```@docs
# Polfed.PolfedCore.Parallelization
# Polfed.MulColsParallel
# Polfed.TwoLevelParallel
# Polfed.NoParallel
# ```
#
# ## [Processing Unit Types](@id docs_processing_units)
#
# ```@docs
# Polfed.PolfedCore.CPU
# ```
#
# GPU processing is available through `GPU()` in CUDA-enabled environments.
#
# ## Factorization-Related Strategy Types
#
# ```@docs
# Polfed.Lanczos.FullRO
# Polfed.Lanczos.PartialRO
# Polfed.Lanczos.MatrixBasis
# Polfed.Lanczos.HybridMatrixBasis
# ```
#
# ## Supported `target` Specifications
#
# - `:maxdos`
# - `:middle`
# - `(:offset, frac)`
# - `(:unrescaled, E)` or plain `E::Real`
# - `(:rescaled, e)`
