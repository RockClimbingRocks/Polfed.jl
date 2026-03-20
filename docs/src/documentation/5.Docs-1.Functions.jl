# # [Documentation for Polfed.jl](@id docs_main_reference)
#
# This page documents POLFED solver entry points used throughout the tutorials.
#
# ## Main Solver Entry Point
#
# Use [`polfed`](@ref Polfed.polfed) with:
# - matrix input: `polfed(mat, x0, howmany, target; ...)`
# - mapping callback: `polfed(f!, x0, howmany, target; ...)`
#
# ```@docs
# Polfed.polfed
# ```
#
# ## Mapping-Optimization Helpers
#
# These helpers are available when constructing high-performance custom mapping
# workflows:
#
# ```@docs
# Polfed.optimized_mapping!
# Polfed.optimized_clenshaw_recurrence_relation!
# Polfed.optimized_clenshaw_final_sum!
# ```
#
# ## Low-Level Factorization Utility
#
# ```@docs
# Polfed.Lanczos.lanczos
# ```
#
# ## Example: Passing Full Configuration into `polfed`
#
# ```julia
# using Polfed
# using Polfed.QSun: quantum_sun_hamiltonian
# using LinearAlgebra
#
# mat = quantum_sun_hamiltonian(12, 2; sparse=true)
# x0 = rand(size(mat, 1)); x0 ./= norm(x0)
# howmany = 100
# target = 0.0
#
# mapping_cfg = MappingConfig(
#     parallel_strategy = MulColsParallel(),
#     optimize_mapping = false,
# )
#
# transform_cfg = TransformConfig(
#     normalization = 10.0,
#     cutoff = 1.7,
#     order_safety_factor = 0.95,
# )
#
# fact_cfg = FactorizationConfig(
#     tol = 1e-15,
#     eigentol = 1e-10,
#     which = :LR,
# )
#
# dos_cfg = DoSConfig(
#     N = 200,
#     R = 300,
# )
#
# vals, vecs, report = polfed(
#     mat,
#     x0,
#     howmany,
#     target;
#     produce_report = true,
#     mapping = mapping_cfg,
#     transform = transform_cfg,
#     fact = fact_cfg,
#     dos = dos_cfg,
# )
# ```
