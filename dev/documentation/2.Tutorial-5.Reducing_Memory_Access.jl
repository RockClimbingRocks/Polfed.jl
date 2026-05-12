# # [Reducing Memory Access](@id Reducing_Memory_Access)
#
# POLFED performance is often limited by memory bandwidth, not FLOPs. Reducing
# unnecessary reads/writes in mapping is therefore one of the strongest
# optimization directions with [`polfed`](@ref Polfed.polfed).
#
# ## Why This Matters
#
# Spectral transformation is usually the most time-consuming part of POLFED.
# Since POLFED is highly memory-bound, reducing memory access in mapping can
# improve the whole algorithm.
#
# Coalesced memory access is also important: contiguous/regular access patterns
# help runtime systems optimize memory traffic and can improve vectorization
# behavior (including SIMD on CPU).
#
# ## Configuration
#
# This path is most useful when your Hamiltonian has structure you can exploit.
# For the beginner Quantum Sun model, there is usually no trivial custom mapping
# that gives large gains. In XXZ model, the structure is richer and so there
# are more optimizations available. See
# [Custom Mapping](@ref tutorial_xxz_custom_mapping) and
# [GPU Implementation](@ref tutorial_xxz_rescaled_clenshaw).
#
# ```julia
# mapping = MappingConfig(
#     parallel_strategy = MulColsParallel(),
#     f!_rescaled = f_rescaled!,
#     Emin = Emin,
#     Emax = Emax,
# )
#
# vals, vecs, report = polfed(f!, x0, howmany, target;
#     produce_report=true,
#     mapping=mapping,
# )
# display_report(report)
# ```
#
# Here we use [`MappingConfig`](@ref Polfed.MappingConfig) with
# [`MulColsParallel`](@ref Polfed.MulColsParallel), then inspect
# [`Report`](@ref Polfed.PolfedCore.Report) via
# [`display_report`](@ref Polfed.display_report).
#
# ## Why Naive Rescaling Can Be Expensive
#
# A naive rescaled mapping can be written as:
#
# ```julia
# f!_rescaled_bad = (Y, X) -> begin
#     f!(Y, X)
#     @. Y *= 1 / spread
#     @. Y -= (center / spread) * X
# end
# ```
#
# This adds extra vector passes and therefore extra memory traffic. A direct
# model-specific rescaled mapping is usually better.
#
# Even a simpler intermediate step can help:
# - build a copied/rescaled matrix `mat_rescaled`,
# - use `mul!(Y, mat_rescaled, X)` as your `f!_rescaled` mapping in
#   [`MappingConfig`](@ref Polfed.MappingConfig).
#
# This often gives a noticeable speedup versus repeatedly applying
# `f!_rescaled_bad`.
