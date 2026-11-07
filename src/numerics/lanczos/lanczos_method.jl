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
are only reordered here. With a [`MatrixBasis`](@ref) the returned `vecs` is a view into
the basis storage, which the extraction reuses (see [`extract_eigenpairs!`](@ref)).
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
    # eigenvectors are stored as columns (ColumnLayout) or rows (RowLayout). They are
    # permuted in place: a sorted copy would be a second full eigenvector matrix.
    perm = sortperm(vals)                       # vals is a host vector (also for a GPU run)

    vals_sorted = vals[perm]
    vecs_sorted = permute_eigenvectors!(iterator.layout, vecs, perm)
    residuals_sorted = residuals[perm]
    converged = count(<(criterion.eigentol), residuals_sorted)

    return LanczosResult(vals_sorted, vecs_sorted, residuals_sorted, converged, report)
end
