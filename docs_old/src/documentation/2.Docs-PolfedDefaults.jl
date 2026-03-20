
# # `PolfedDefaults` Module Reference
#
# This page provides a detailed reference for all the default settings
# available in the `PolfedDefaults` submodule. These constants and functions
# provide the baseline behavior for the `polfed` function and its underlying
# algorithms.
#
# While these defaults are chosen to be sensible for a wide range of problems,
# they can be overridden by passing configuration objects like `MappingConfig`,
# `TransformConfig`, `FactorizationConfig`, and `DoSConfig` to the main `polfed`
# function.

# ```@docs
# Polfed.PolfedDefaults
# ```

# ## Krylov Dimension Prediction
#
# Settings related to estimating the required dimension of the Krylov subspace.

# ```@docs
# Polfed.PolfedDefaults.expectedkrylovdim
# ```

# ## Reporting Options
#
# Top-level flags that control reporting and optimization behavior, plus verbosity.

# ```@docs
# Polfed.PolfedDefaults.produce_report
# Polfed.PolfedDefaults.optimize_mapping
# Polfed.PolfedDefaults.POLFED_SILENT_LEVEL
# Polfed.PolfedDefaults.POLFED_WARN_LEVEL
# Polfed.PolfedDefaults.POLFED_INFO_LEVEL
# Polfed.PolfedDefaults.POLFED_DEBUG_LEVEL
# Polfed.PolfedDefaults.verbosity
# ```

# ## Spectral Transformation Defaults
#
# Default parameters for the polynomial spectral transformation, which is the core
# filtering mechanism in POLFED. These values are used by `TransformConfig` unless
# overridden.

# ```@docs
# Polfed.PolfedDefaults.coefficients
# Polfed.PolfedDefaults.polynomialtype
# Polfed.PolfedDefaults.cutoff
# Polfed.PolfedDefaults.normalization
# Polfed.PolfedDefaults.order_safety_factor
# Polfed.PolfedDefaults.parallelization
# Polfed.PolfedDefaults.overestimate_iters
# ```

# ## Lanczos Defaults
#
# Default parameters for the Lanczos algorithm, which is used to diagonalize
# the filtered operator.

# ```@docs
# Polfed.PolfedDefaults.rot
# Polfed.PolfedDefaults.basistype
# Polfed.PolfedDefaults.tol
# Polfed.PolfedDefaults.eigentol
# Polfed.PolfedDefaults.which
# ```

# ## Density of States Defaults
#
# Default parameters for calculating the Density of States (DoS).

# ```@docs
# Polfed.PolfedDefaults.kernel
# Polfed.PolfedDefaults.N
# Polfed.PolfedDefaults.R
# ```
