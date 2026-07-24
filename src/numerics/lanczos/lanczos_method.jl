"""
    cleanup_after_lanczos!(pu::ProcessingUnit; collect_garbage::Bool=true) -> nothing

Release Lanczos working memory. On GPU this synchronizes and reclaims the device
pool. `collect_garbage` triggers a full `GC.gc(true)`; pass `false` for the many
small extremal probes where a forced collection per call is wasteful.
"""
function cleanup_after_lanczos!(pu::ProcessingUnit; collect_garbage::Bool=true)
    if pu isa GPU
        gpu_synchronize()
        gpu_reclaim()
    end
    collect_garbage && GC.gc(true)
    return nothing
end

"""
    lanczos_method(iterator, criterion, state, basistype, pu; collect_garbage=true) -> LanczosResult

Build the Krylov basis, run the Lanczos iterations, and return the converged
eigenpairs (sorted ascending) as a [`LanczosResult`](@ref). The eigenpair count is
dynamic (value-based targets), so the outputs come straight from the extraction and
are only reordered here.
"""
function lanczos_method(
    iterator::LanczosIterator,
    criterion::ConvergenceCriterion,
    state::ConvergenceState,
    basistype::Type{<:OrthonormalBasis},
    pu::ProcessingUnit;
    collect_garbage::Bool=true,
)
    krylovbasis = createbasis(criterion.maxdim, iterator.x0, basistype, pu, iterator.layout)
    vals, vecs, residuals, report = lanczos_algorithm(iterator, criterion, state, krylovbasis, pu)
    cleanup_after_lanczos!(pu; collect_garbage=collect_garbage)

    nconv = length(residuals)
    if nconv <= 1
        converged = count(<(criterion.eigentol), residuals)
        return LanczosResult(vals, vecs, residuals, converged, report)
    end

    # sort the converged eigenpairs ascending and align vecs + per-eigenpair residuals.
    # eigenvectors are stored as columns (ColumnLayout) or rows (RowLayout).
    vals_cpu = is_gpu_array(vals) ? collect(vals) : vals
    perm = sortperm(vals_cpu)

    vals_sorted = pu.Vector(vals[perm])
    vecs_sorted = iterator.layout isa RowLayout ? vecs[perm, :] : vecs[:, perm]
    residuals_sorted = residuals[perm]
    converged = count(<(criterion.eigentol), residuals_sorted)

    return LanczosResult(vals_sorted, vecs_sorted, residuals_sorted, converged, report)
end
