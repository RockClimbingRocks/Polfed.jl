# Shared preparation for all matrix entrypoints. The operator-building primitives live with the
# mapping slice; this helper only orchestrates them for the public row-native paths.

"""
    _prepare_matrix_row_pipeline(H, x0, mc) -> (f!, x0_row, mc_row)

Prepare a concrete matrix for any row-native POLFED entrypoint. The helper resolves missing
spectral bounds, builds the selected row operator, reorients `x0`, and returns a mapping config
that carries the row-specific rescaled operator. Count, window, patched, and full entrypoints
then differ only in the pipeline they call.
"""
function _prepare_matrix_row_pipeline(H::AbstractMatrix, x0::AbstractVecOrMat, mc::MappingConfig)
    size(H, 1) == size(H, 2) ||
        throw(DimensionMismatch("Matrix POLFED entrypoints require a square matrix."))

    # x0 orientation: an AbstractMatrix with block_size ≤ dim is already row form (block_size × dim);
    # a column block (dim × block_size) or a vector is not. Used for the probe seed and the reorient.
    is_row = x0 isa AbstractMatrix && size(x0, 1) <= size(x0, 2)

    # Spectral bounds: a single column probe on H (unless the config supplies both). The seed is one
    # length-dim vector — the first column of a column block, the first row of a row block.
    if isnothing(mc.Emin) || isnothing(mc.Emax)
        v0 = x0 isa AbstractVector ? x0 : is_row ? view(x0, 1, :) : view(x0, :, 1)
        E1, E2 = lanczos_extrema(H; x0 = collect(v0))
        Emin = isnothing(mc.Emin) ? E1 : mc.Emin
        Emax = isnothing(mc.Emax) ? E2 : mc.Emax
    else
        Emin, Emax = mc.Emin, mc.Emax
    end
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    # Row-layout operators (fast value-grouped kernels, or generic mul! fallback).
    bs = x0 isa AbstractVector ? 1 : minimum(size(x0))
    rowops = _build_matrix_rowops(H, mc.optimize_mapping, bs, a, b)

    # x0 → row orientation (block_size × dim); reorient only a column block or a vector (→ 1×D).
    x0_row = is_row ? x0 : permutedims(x0)

    mc_row = MappingConfig(optimize_mapping = mc.optimize_mapping,
                           f!_rescaled       = rowops.f!_rescaled,
                           parallel_strategy = NoParallel(),
                           Emin = Emin, Emax = Emax)

    return rowops.f!, x0_row, mc_row
end
