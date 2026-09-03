"""
    run_double_polfed_target_pipeline(f!, x0, howmany, target1, target2; kwargs...)
        -> DoublePolfedResult

Two-window POLFED for two count targets: `howmany` eigenpairs near `target1` **and** `howmany`
near `target2` (or one count per window, when `howmany` is a 2-tuple).

The mapping and the DoS are resolved **once** for both windows; each point target is then grown
into the rescaled interval holding its requested count (exactly as
[`run_polfed_target_pipeline`](@ref) does), and the run takes one of two paths
([`resolve_double_strategy`](@ref)):

- `:double` — ONE filtered Lanczos run with a [`DoubleDeltaFilter`](@ref) and a
  `BothEnds(NumEigs, NumEigs)` target; the pooled result is split into the two windows.
- `:separate` — two ordinary single-filter count runs, sequentially (the matvec is already
  parallel; two competing Lanczos runs would not help).

Units follow [`run_polfed_target_pipeline`](@ref): a count criterion needs no rescaled window, so
Ritz values are measured against the unrescaled `mp.f!` and `eigentol` is used as given.

# Arguments
- `howmany`: an `Integer` (that many from EACH window) or a 2-tuple `(n₁, n₂)`, in the caller's
  argument order.
- `target1`, `target2`: **point** specs (`:maxdos`, a plain energy, `(:rescaled, v)`, …), in
  either order; a window spec here is an error — use the window method.

# Keywords
- `strategy::Symbol`: `:auto`, `:double`, or `:separate`.
- `mc`, `dosc`, `tol`, `eigentol`, `overestimate_iters`, `layout`: as in [`polfed`](@ref).

# Returns
- [`DoublePolfedResult`](@ref): the two [`PolfedResult`](@ref)s in the caller's argument order,
  plus the strategy decision.

# Throws
- `ArgumentError`: a window spec, a non-positive count, or two targets whose DoS-grown intervals
  overlap (a single connected region — use [`polfed`](@ref)).
"""
function run_double_polfed_target_pipeline(
    f!,
    x0::AbstractVecOrMat,
    howmany,
    target1,
    target2;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.double_polfed.tol,
    eigentol::Real = Defaults.double_polfed.eigentol,
    overestimate_iters::Real = Defaults.double_polfed.overestimate_iters,
    strategy::Symbol = Defaults.double_polfed.strategy,
    layout::Layout = ColumnLayout(),
)
    timer = StageTimer()
    howmany_a, howmany_b = _double_normalize_howmany(howmany)
    spec_a = normalize_target_spec(target1)
    spec_b = normalize_target_spec(target2)
    (spec_a isa TargetWindowSpec || spec_b isa TargetWindowSpec) && throw(ArgumentError(
        "The double count pipeline requires two POINT targets, not windows. For window targets " *
        "use `double_polfed(f!, x0, target1, target2)`.",
    ))

    polfed_log(POLFED_INFO_LEVEL, "double_polfed started (count mode).",
               target1 = format_target_strategy(spec_a), target2 = format_target_strategy(spec_b),
               howmany = (howmany_a, howmany_b), strategy = strategy,
               tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    dos = @stage timer :dos calculate_dos(dosc, dos_operator(mp), mp.sys)

    center_a = @stage timer :interval resolve_target(spec_a, howmany_a, dos, mp.a, mp.b)
    center_b = @stage timer :interval resolve_target(spec_b, howmany_b, dos, mp.a, mp.b)
    interval_a = @stage timer :interval resolve_interval(center_a, dos)
    interval_b = @stage timer :interval resolve_interval(center_b, dos)

    lower, upper, swapped = _double_order_windows(interval_a, interval_b)
    spec_lower, spec_upper = swapped ? (spec_b, spec_a) : (spec_a, spec_b)
    n_lower, n_upper = swapped ? (howmany_b, howmany_a) : (howmany_a, howmany_b)
    # The count selection picks the pairs nearest each window's target, so the targets have to be
    # carried through in true energy — and in the same lower/upper order as everything else here.
    center_lower, center_upper = swapped ? (center_b, center_a) : (center_a, center_b)
    target_lower = mp.a * center_lower.rescaled_energy + mp.b
    target_upper = mp.a * center_upper.rescaled_energy + mp.b

    strat = @stage timer :filter resolve_double_strategy(lower, upper, strategy)

    if strat.mode === :double
        lower_result, upper_result = _double_joint_count(
            mp, strat.double_filter, lower, upper, spec_lower, spec_upper,
            n_lower, n_upper, tol, eigentol, overestimate_iters, layout, timer;
            target_lower = target_lower, target_upper = target_upper,
        )
    else
        lower_result = _double_separate_count_run(
            mp, lower, spec_lower, strat.filter1, n_lower, tol, eigentol,
            overestimate_iters, layout, _double_run_timer(timer); label = "lower window run",
            target_energy = target_lower,
        )
        upper_result = _double_separate_count_run(
            mp, upper, spec_upper, strat.filter2, n_upper, tol, eigentol,
            overestimate_iters, layout, _double_run_timer(timer); label = "upper window run",
            target_energy = target_upper,
        )
    end

    return _double_result(strat, lower_result, upper_result, swapped)
end
