# Public double-window POLFED entrypoints. Same shape as polfed.jl: one method per
# (operator|matrix) × (count|window) combination, each thin over its pipeline. The two modes are
# distinguished by ARITY — 4 positional arguments is the window mode, 5 is the count mode (the
# extra one being `howmany`) — the same convention `patched_polfed` uses. The two targets are
# separate positional arguments rather than a tuple because a bare `(lo, hi)` is already a valid
# window spec, so a tuple of targets would be genuinely ambiguous.

"""
    double_polfed(f!, x0, howmany, target1, target2; strategy, mc, dosc, tol, eigentol,
                  overestimate_iters, verbosity) -> DoublePolfedResult

Two-window POLFED on an operator: compute `howmany` eigenpairs near `target1` **and** `howmany`
near `target2`, sharing one spectral-bound probe and one DoS estimate.

Two ways to serve two windows, and this entrypoint picks between them:

- one [`DoubleDeltaFilter`](@ref) run — a single polynomial with two peaks (`+1` on the lower
  window, `−1` on the upper), so ONE Krylov space converges both windows at its two ends;
- two ordinary [`polfed`](@ref) runs, one per window.

The joint filter's cost per matvec is `K_d = max(K₁, K₂)` — both windows pay the sharper peak's
order — so it only pays off when the two windows need comparable orders. With
`r = min(K₁,K₂)/max(K₁,K₂)`, the default `strategy = :auto` takes the joint run when
`r ≥ `[`Defaults.double_polfed.double_ratio`](@ref) and two runs otherwise; `:double`/`:separate`
force a path (for benchmarking). A forced `:double` still falls back to two runs if the joint
filter cannot be built — e.g. a window touching a spectrum edge, which needs an [`EdgeFilter`](@ref).

# Arguments
- `x0::AbstractVecOrMat`: start vector (block size 1) or block, fixing dimension, element type and
  device — as in [`polfed`](@ref).
- `howmany`: an `Integer` (that many eigenpairs from EACH window) or a 2-tuple `(n₁, n₂)`.
- `target1`, `target2`: **point** target specs (see [`normalize_target_spec`](@ref)), in either
  order — the result comes back in the order given.

# Returns
- [`DoublePolfedResult`](@ref): two [`PolfedResult`](@ref)s, iterable/destructurable as
  `r1, r2 = double_polfed(…)`, carrying the strategy decision and both polynomial orders.

Thin wrapper over [`run_double_polfed_target_pipeline`](@ref), in the default column layout — the
caller's `f!`/`x0` orientation is left untouched.
"""
double_polfed(f!, x0::AbstractVecOrMat, howmany, target1, target2;
              strategy::Symbol         = Defaults.double_polfed.strategy,
              mc::MappingConfig        = MappingConfig(),
              dosc::DoSConfig          = DoSConfig(),
              tol::Real                = Defaults.double_polfed.tol,
              eigentol::Real           = Defaults.double_polfed.eigentol,
              overestimate_iters::Real = Defaults.double_polfed.overestimate_iters,
              verbosity::Union{Nothing,Integer} = nothing) =
    @with_verbosity verbosity run_double_polfed_target_pipeline(f!, x0, howmany, target1, target2;
                 mc = mc, dosc = dosc, tol = tol, eigentol = eigentol,
                 overestimate_iters = overestimate_iters, strategy = strategy)  # column layout

"""
    double_polfed(H::AbstractMatrix, x0, howmany, target1, target2; strategy, mc, dosc, tol,
                  eigentol, overestimate_iters, verbosity) -> DoublePolfedResult

Matrix (count-based) two-window entrypoint: as [`double_polfed(f!, x0, howmany, target1,
target2)`](@ref), on the fast **row-layout** path. Owns `H`, so it probes `Emin`/`Emax`, builds the
value-grouped row kernels and reorients `x0` ([`_prepare_matrix_row_pipeline`](@ref)) before
delegating with `layout = RowLayout()`.
"""
function double_polfed(H::AbstractMatrix, x0::AbstractVecOrMat, howmany, target1, target2;
                       strategy::Symbol         = Defaults.double_polfed.strategy,
                       mc::MappingConfig        = MappingConfig(),
                       dosc::DoSConfig          = DoSConfig(),
                       tol::Real                = Defaults.double_polfed.tol,
                       eigentol::Real           = Defaults.double_polfed.eigentol,
                       overestimate_iters::Real = Defaults.double_polfed.overestimate_iters,
                       verbosity::Union{Nothing,Integer} = nothing)

    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_double_polfed_target_pipeline(f!, x0_row, howmany, target1, target2;
                        mc = mc_row, dosc = dosc_row, tol = tol, eigentol = eigentol,
                        overestimate_iters = overestimate_iters, strategy = strategy,
                        layout = RowLayout())
end

"""
    double_polfed(f!, x0, target1, target2; strategy, mc, dosc, tol, eigentol,
                  overestimate_iters, verbosity) -> DoublePolfedResult

Window-target two-window POLFED on an operator: compute **all** eigenpairs whose true energy lies
in `target1` and all whose energy lies in `target2`. No `howmany` is passed — each window defines
its own set, sized from the DoS only to budget the Krylov space (as in the single-window
[`polfed(f!, x0, target)`](@ref)).

`target1` and `target2` must both be **window** specs — `(lo, hi)`, `(:rescaled, (lo, hi))`, or
`(:unrescaled, (lo, hi))` — given in either order; a point target here is an error (use the
5-argument count form). The windows must be disjoint with a gap between them; one connected region
is a job for [`polfed`](@ref) or [`patched_polfed`](@ref).

The joint-versus-separate decision, the `strategy` override and the return value are exactly as in
the count-based method above. Thin wrapper over [`run_double_polfed_window_pipeline`](@ref).
"""
double_polfed(f!, x0::AbstractVecOrMat, target1, target2;
              strategy::Symbol         = Defaults.double_polfed.strategy,
              mc::MappingConfig        = MappingConfig(),
              dosc::DoSConfig          = DoSConfig(),
              tol::Real                = Defaults.double_polfed.tol,
              eigentol::Real           = Defaults.double_polfed.eigentol,
              overestimate_iters::Real = Defaults.double_polfed.overestimate_iters,
              verbosity::Union{Nothing,Integer} = nothing) =
    @with_verbosity verbosity run_double_polfed_window_pipeline(f!, x0, target1, target2;
                 mc = mc, dosc = dosc, tol = tol, eigentol = eigentol,
                 overestimate_iters = overestimate_iters, strategy = strategy)  # column layout

"""
    double_polfed(H::AbstractMatrix, x0, target1, target2; strategy, mc, dosc, tol, eigentol,
                  overestimate_iters, verbosity) -> DoublePolfedResult

Matrix **window-target** two-window entrypoint: all eigenpairs of `H` inside each of the two
windows, on the fast row-layout path. Same row-prep as the count-based matrix method; delegates to
[`run_double_polfed_window_pipeline`](@ref).
"""
function double_polfed(H::AbstractMatrix, x0::AbstractVecOrMat, target1, target2;
                       strategy::Symbol         = Defaults.double_polfed.strategy,
                       mc::MappingConfig        = MappingConfig(),
                       dosc::DoSConfig          = DoSConfig(),
                       tol::Real                = Defaults.double_polfed.tol,
                       eigentol::Real           = Defaults.double_polfed.eigentol,
                       overestimate_iters::Real = Defaults.double_polfed.overestimate_iters,
                       verbosity::Union{Nothing,Integer} = nothing)

    f!, x0_row, mc_row, dosc_row = _prepare_matrix_row_pipeline(H, x0, mc, dosc)
    return @with_verbosity verbosity run_double_polfed_window_pipeline(f!, x0_row, target1, target2;
                        mc = mc_row, dosc = dosc_row, tol = tol, eigentol = eigentol,
                        overestimate_iters = overestimate_iters, strategy = strategy,
                        layout = RowLayout())
end
