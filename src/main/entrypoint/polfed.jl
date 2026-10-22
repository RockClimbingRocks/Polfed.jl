# Public POLFED entrypoints, one method per (operator|matrix) × (count|window) combination. All are
# thin: they prepare inputs and call that mode's pipeline. Count-based methods take `howmany` and a
# point/region `target` → `run_polfed_target_pipeline`; window methods take a window `target` and no
# `howmany` → `run_polfed_window_pipeline`. Operator methods run column-native on the caller's `f!`;
# matrix methods own `H` and go row-native via `_prepare_matrix_row_pipeline` (entrypoint_helpers.jl).

"""
    polfed(f!, x0, howmany, target; mc, dosc, tol, eigentol, overestimate_iters, verbosity) -> PolfedResult

Polynomial-Filtered Exact Diagonalization on an operator: compute `howmany` eigenpairs of `f!`
(in-place action `f!(Y, X)` computing `Y ← H·X`) near `target`, via polynomial-filtered Lanczos.

`x0::AbstractVecOrMat` is the start vector (block size 1) or block (block-Lanczos) and fixes the
dimension, element type, and device. `target` is normalized by [`normalize_target_spec`](@ref)
(`:maxdos`, `:middle`, a plain energy, `(:rescaled, v)`, `(:offset, η)`, …).

Returns a [`PolfedResult`](@ref): the eigenpairs (true energies of `H`) **filtered to the target
interval** and sorted ascending, iterable as `(val, vec)` pairs, carrying the structured
[`Report`](@ref). A `@warn` (not an error) is emitted if fewer than `howmany` land in-interval.

`verbosity` (default `nothing`) raises the POLFED log level for this call only and restores it
afterwards — `3` traces every stage decision, including the full pre-Lanczos setup. Every POLFED
entrypoint takes it; see [`with_verbosity`](@ref) for the scoped form.

Thin wrapper over [`run_polfed_target_pipeline`](@ref), in the default column layout — a caller's
`f!`/`x0` orientation is left untouched.
"""
polfed(f!, x0::AbstractVecOrMat, howmany::Integer, target;
       mc::MappingConfig        = MappingConfig(),
       dosc::DoSConfig          = DoSConfig(),
       tol::Real                = Defaults.polfed.tol,
       eigentol::Real           = Defaults.polfed.eigentol,
       overestimate_iters::Real = Defaults.polfed.overestimate_iters,
       verbosity::Union{Nothing,Integer} = nothing) =
    @with_verbosity verbosity run_polfed_target_pipeline(f!, x0, howmany, target;
                 mc = mc, dosc = dosc, tol = tol, eigentol = eigentol,
                 overestimate_iters = overestimate_iters)   # column layout (default)

"""
    polfed(H::AbstractMatrix, x0, howmany, target; mc, dosc, tol, eigentol, overestimate_iters, verbosity) -> PolfedResult

Matrix (count-based) entrypoint: compute `howmany` eigenpairs of the concrete Hermitian matrix `H`
near `target`, using the fast **row-layout** path. Owns `H`, so it probes `Emin`/`Emax`, builds the
value-grouped row kernels, and reorients `x0` to row form ([`_prepare_matrix_row_pipeline`](@ref))
before delegating to [`run_polfed_target_pipeline`](@ref) with `layout = RowLayout()`.

`mc.optimize_mapping` selects the row kernel (`false` | `:auto`/`true` | a kernel `Symbol` | a
`(kernel=…, G=…, W=…)` NamedTuple). Returns a [`PolfedResult`](@ref).
"""
function polfed(H::AbstractMatrix, x0::AbstractVecOrMat, howmany::Integer, target;
                mc::MappingConfig        = MappingConfig(),
                dosc::DoSConfig          = DoSConfig(),
                tol::Real                = Defaults.polfed.tol,
                eigentol::Real           = Defaults.polfed.eigentol,
                overestimate_iters::Real = Defaults.polfed.overestimate_iters,
                verbosity::Union{Nothing,Integer} = nothing)

    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_polfed_target_pipeline(f!, x0_row, howmany, target;
                        mc = mc_row, dosc = dosc_row, tol = tol, eigentol = eigentol,
                        overestimate_iters = overestimate_iters, layout = RowLayout())
end








"""
    polfed(f!, x0, target; mc, dosc, tol, eigentol, overestimate_iters, verbosity) -> PolfedResult

Window-target POLFED on an operator: compute **all** eigenpairs of `f!` whose true energy lies in
the target window `target`, via polynomial-filtered Lanczos. Unlike the count-based
[`polfed(f!, x0, howmany, target)`](@ref), no `howmany` is passed — the window itself defines the
set. Its size is *estimated* from the DoS (to budget the Krylov space), and Lanczos stops once every
in-window eigenpair has converged (a [`ToValue`](@ref) boundary criterion at the filter cutoff), so
the exact count is discovered, not requested.

`target` must be a **window** spec (see [`normalize_target_spec`](@ref)): `(:rescaled, (lo, hi))`,
`(:unrescaled, (lo, hi))`, or a bare `(lo, hi)` (unrescaled by default). A point target here is an
error — use the 4-arg count-based form for that.

Thin wrapper over [`run_polfed_window_pipeline`](@ref), in the default column layout.
"""
polfed(f!, x0::AbstractVecOrMat, target;
       mc::MappingConfig        = MappingConfig(),
       dosc::DoSConfig          = DoSConfig(),
       tol::Real                = Defaults.polfed.tol,
       eigentol::Real           = Defaults.polfed.eigentol,
       overestimate_iters::Real = Defaults.polfed.overestimate_iters,
       verbosity::Union{Nothing,Integer} = nothing) =
    @with_verbosity verbosity run_polfed_window_pipeline(f!, x0, target;
                 mc = mc, dosc = dosc, tol = tol, eigentol = eigentol,
                 overestimate_iters = overestimate_iters)   # column layout (default)




"""
    polfed(H::AbstractMatrix, x0, target; mc, dosc, tol, eigentol, overestimate_iters, verbosity) -> PolfedResult

Matrix **window-target** entrypoint: compute all eigenpairs of `H` whose true energy lies in the
window `target`, using the fast row-layout path. Same row-prep as the count-based matrix method
([`_prepare_matrix_row_pipeline`](@ref)); delegates to [`run_polfed_window_pipeline`](@ref) instead.
`target` must be a window spec — see [`polfed(f!, x0, target)`](@ref).
"""
function polfed(H::AbstractMatrix, x0::AbstractVecOrMat, target;
                mc::MappingConfig        = MappingConfig(),
                dosc::DoSConfig          = DoSConfig(),
                tol::Real                = Defaults.polfed.tol,
                eigentol::Real           = Defaults.polfed.eigentol,
                overestimate_iters::Real = Defaults.polfed.overestimate_iters,
                verbosity::Union{Nothing,Integer} = nothing)

    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_polfed_window_pipeline(f!, x0_row, target;
                        mc = mc_row, dosc = dosc_row, tol = tol, eigentol = eigentol,
                        overestimate_iters = overestimate_iters, layout = RowLayout())
end
