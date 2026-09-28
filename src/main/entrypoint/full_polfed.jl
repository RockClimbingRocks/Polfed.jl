# Public full-spectrum POLFED entrypoints. There is no target mode: the whole rescaled
# spectrum `[-1, 1]` is resolved edge-first and stored directly in one dense result matrix.

"""
    full_polfed(f!, x0, howmany_per_patch; kwargs...) -> (vals, vecs, report)

Resolve the complete spectrum of an operator with edge-first patched POLFED.
`howmany_per_patch` controls the approximate number of states in each interior patch.
The result uses one dense `D × D` column-layout eigenvector matrix and returns a
[`PatchedReport`](@ref). If completeness cannot be certified, the converged pairs are
returned with a warning.
"""
function full_polfed(
    f!,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer;
    mc::MappingConfig = MappingConfig(
        optimize_mapping = Defaults.full_polfed.optimize_mapping,
    ),
    dosc::DoSConfig = DoSConfig(
        N = Defaults.full_polfed.N,
        R = Defaults.full_polfed.R,
        kernel = Defaults.full_polfed.kernel,
    ),
    tol::Real = Defaults.full_polfed.tol,
    eigentol::Real = Defaults.full_polfed.eigentol,
    overestimate_iters::Real = Defaults.full_polfed.overestimate_iters,
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.full_polfed.howmany_pad,
    max_retries::Integer = Defaults.full_polfed.max_glue_retries,
    retry_growth::Real = Defaults.full_polfed.retry_budget_growth,
    verbosity::Union{Nothing,Integer} = nothing,
)
    return @with_verbosity verbosity run_full_polfed_pipeline(
        f!, x0, howmany_per_patch;
        mc, dosc, tol, eigentol, overestimate_iters,
        edge_distance, howmany_pad, max_retries, retry_growth,
    )
end

"""
    full_polfed(H, x0, howmany_per_patch; kwargs...) -> (vals, vecs, report)

Matrix form of full POLFED. It prepares the row-native matrix operator and writes
eigenvectors directly into one `D × D` row-layout result matrix.
"""
function full_polfed(
    H::AbstractMatrix,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer;
    mc::MappingConfig = MappingConfig(
        optimize_mapping = Defaults.full_polfed.optimize_mapping,
    ),
    dosc::DoSConfig = DoSConfig(
        N = Defaults.full_polfed.N,
        R = Defaults.full_polfed.R,
        kernel = Defaults.full_polfed.kernel,
    ),
    tol::Real = Defaults.full_polfed.tol,
    eigentol::Real = Defaults.full_polfed.eigentol,
    overestimate_iters::Real = Defaults.full_polfed.overestimate_iters,
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.full_polfed.howmany_pad,
    max_retries::Integer = Defaults.full_polfed.max_glue_retries,
    retry_growth::Real = Defaults.full_polfed.retry_budget_growth,
    verbosity::Union{Nothing,Integer} = nothing,
)
    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_full_polfed_pipeline(
        f!, x0_row, howmany_per_patch;
        mc = mc_row, dosc = dosc_row, tol, eigentol, overestimate_iters,
        layout = RowLayout(),
        edge_distance, howmany_pad, max_retries, retry_growth,
    )
end
