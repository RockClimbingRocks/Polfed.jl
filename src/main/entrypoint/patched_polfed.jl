# Public patched POLFED entrypoints. Count and window modes differ by arity:
# count mode receives `howmany` plus `howmany_per_patch`; window mode receives only
# `howmany_per_patch`. Operator inputs stay column-native, while matrix inputs use
# `_prepare_matrix_row_pipeline` and force `RowLayout()`.

"""
    patched_polfed(f!, x0, howmany, howmany_per_patch, target; kwargs...)
        -> (vals, vecs, report)

Compute the `howmany` eigenpairs nearest a point target with edge-first patched POLFED.
`howmany_per_patch` controls the approximate interior patch size. The target accepts the
same point specifications as [`polfed`](@ref), but not an explicit window.

The operator `f!` must write `H * X` into `Y` when called as `f!(Y, X)`. Results are
sorted by eigenvalue; `vecs` is one preallocated column-layout matrix and `report` is a
[`PatchedReport`](@ref).
"""
function patched_polfed(
    f!,
    x0::AbstractVecOrMat,
    howmany::Integer,
    howmany_per_patch::Integer,
    target;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.polfed.tol,
    eigentol::Real = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters,
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.patched_polfed.howmany_pad,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
    verbosity::Union{Nothing,Integer} = nothing,
)
    return @with_verbosity verbosity run_patched_polfed_target_pipeline(
        f!, x0, howmany, howmany_per_patch, target;
        mc, dosc, tol, eigentol, overestimate_iters,
        edge_distance, howmany_pad, max_retries, retry_growth,
    )
end

"""
    patched_polfed(H, x0, howmany, howmany_per_patch, target; kwargs...)
        -> (vals, vecs, report)

Matrix form of patched count-target POLFED. It prepares optimized row-native operator
kernels, converts `x0` to row layout, and returns eigenvectors as a `howmany × D` matrix.
"""
function patched_polfed(
    H::AbstractMatrix,
    x0::AbstractVecOrMat,
    howmany::Integer,
    howmany_per_patch::Integer,
    target;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.polfed.tol,
    eigentol::Real = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters,
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.patched_polfed.howmany_pad,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
    verbosity::Union{Nothing,Integer} = nothing,
)
    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_patched_polfed_target_pipeline(
        f!, x0_row, howmany, howmany_per_patch, target;
        mc = mc_row, dosc = dosc_row, tol, eigentol, overestimate_iters,
        layout = RowLayout(),
        edge_distance, howmany_pad, max_retries, retry_growth,
    )
end

"""
    patched_polfed(f!, x0, howmany_per_patch, target; kwargs...)
        -> (vals, vecs, report)

Compute every eigenpair in an explicit spectral window with edge-first patched POLFED.
The target must be `(:rescaled, (left, right))`, `(:unrescaled, (left, right))`, or a
bare unrescaled `(left, right)` tuple. Each returned eigenvector is an independent vector,
so completed patch matrices can be released immediately.
"""
function patched_polfed(
    f!,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer,
    target;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.polfed.tol,
    eigentol::Real = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters,
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.patched_polfed.howmany_pad,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
    verbosity::Union{Nothing,Integer} = nothing,
)
    return @with_verbosity verbosity run_patched_polfed_window_pipeline(
        f!, x0, howmany_per_patch, target;
        mc, dosc, tol, eigentol, overestimate_iters,
        edge_distance, howmany_pad, max_retries, retry_growth,
    )
end

"""
    patched_polfed(H, x0, howmany_per_patch, target; kwargs...)
        -> (vals, vecs, report)

Matrix form of patched window POLFED. It uses the row-native matrix path and returns a
vector of independently owned eigenvectors in the requested window.
"""
function patched_polfed(
    H::AbstractMatrix,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer,
    target;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.polfed.tol,
    eigentol::Real = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters,
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.patched_polfed.howmany_pad,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
    verbosity::Union{Nothing,Integer} = nothing,
)
    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_patched_polfed_window_pipeline(
        f!, x0_row, howmany_per_patch, target;
        mc = mc_row, dosc = dosc_row, tol, eigentol, overestimate_iters,
        layout = RowLayout(),
        edge_distance, howmany_pad, max_retries, retry_growth,
    )
end
