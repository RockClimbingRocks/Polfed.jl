# # [Reports, Logging, and Defaults](@id docs_report_defaults)
#
# Reporting is enabled through `produce_report=true` in
# [`polfed`](@ref Polfed.polfed). The returned
# [`Report`](@ref Polfed.PolfedCore.Report) provides spectral-transform,
# factorization, and benchmark diagnostics.
#
# ## Report Types and Display Functions
#
# ```@docs
# Polfed.display_report
# Polfed.PolfedCore.Report
# Polfed.Lanczos.FactorizationReport
# Polfed.Lanczos.display_factorization_report
# Polfed.PolfedCore.SpectralTransformReport
# Polfed.PolfedCore.display_spectral_report
# Polfed.PolfedCore.BenchmarkReport
# Polfed.PolfedCore.display_benchmark_report
# ```
#
# ## [Logging Levels and Helpers](@id docs_logging_defaults)
#
# ```@docs
# Polfed.Common.POLFED_SILENT_LEVEL
# Polfed.Common.POLFED_WARN_LEVEL
# Polfed.Common.POLFED_INFO_LEVEL
# Polfed.Common.POLFED_DEBUG_LEVEL
# Polfed.Common.verbosity
# Polfed.Common.should_log
# Polfed.Common.polfed_log
# ```
#
# ## `PolfedDefaults` Module
#
# ```@docs
# Polfed.PolfedCore.PolfedDefaults
# ```
#
# ### Krylov Dimension Prediction
#
# ```@docs
# Polfed.PolfedCore.PolfedDefaults.expectedkrylovdim
# ```
#
# ### Reporting and Mapping Defaults
#
# ```@docs
# Polfed.PolfedCore.PolfedDefaults.produce_report
# Polfed.PolfedCore.PolfedDefaults.optimize_mapping
# ```
#
# ### Spectral Transformation Defaults
#
# ```@docs
# Polfed.PolfedCore.PolfedDefaults.coefficients
# Polfed.PolfedCore.PolfedDefaults.polynomialtype
# Polfed.PolfedCore.PolfedDefaults.cutoff
# Polfed.PolfedCore.PolfedDefaults.normalization
# Polfed.PolfedCore.PolfedDefaults.order_safety_factor
# Polfed.PolfedCore.PolfedDefaults.parallel_strategy
# Polfed.PolfedCore.PolfedDefaults.overestimate_iters
# ```
#
# ### Factorization Defaults
#
# ```@docs
# Polfed.PolfedCore.PolfedDefaults.rot
# Polfed.PolfedCore.PolfedDefaults.basistype
# Polfed.PolfedCore.PolfedDefaults.tol
# Polfed.PolfedCore.PolfedDefaults.eigentol
# Polfed.PolfedCore.PolfedDefaults.which
# ```
#
# ### DoS Defaults
#
# ```@docs
# Polfed.PolfedCore.PolfedDefaults.kernel
# Polfed.PolfedCore.PolfedDefaults.N
# Polfed.PolfedCore.PolfedDefaults.R
# ```
