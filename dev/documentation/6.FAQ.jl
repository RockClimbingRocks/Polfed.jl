# # FAQ / Troubleshooting
#
# ## I requested many eigenpairs and runtime increased. Is this expected?
#
# Yes. `howmany` affects polynomial/filter and factorization behavior. Check
# reporting output to inspect order and iteration balance.
#
# ## Why did factorization type change?
#
# It depends on `x0`:
#
# - vector `x0` -> Lanczos
# - matrix `x0` -> Block Lanczos
#
# ## Which mapping keyword should I use?
#
# Use `parallel_strategy` inside [`MappingConfig`](@ref Polfed.MappingConfig).
#
# ## My run is slow. What should I test first?
#
# 1. `produce_report=true` and inspect mapping/factorization timings with
#    [`display_report`](@ref Polfed.display_report).
# 2. Try `optimize_mapping=true` in
#    [`MappingConfig`](@ref Polfed.MappingConfig).
# 3. For structured models, build custom mapping and optionally custom rescaled/Clenshaw kernels.
#
# ## How do I debug setup issues?
#
# Increase verbosity:
#
# ```julia
# Polfed.Common.verbosity[] = 3
# ```
