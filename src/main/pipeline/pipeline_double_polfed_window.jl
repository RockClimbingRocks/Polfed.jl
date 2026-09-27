"""
    run_double_polfed_window_pipeline(f!, x0, target1, target2; kwargs...) -> DoublePolfedResult

Two-window POLFED for two explicit spectral windows.

The pipeline resolves the mapping and the DoS **once** for both windows, turns the two targets
into two ordered rescaled intervals, and then takes one of two paths
([`resolve_double_strategy`](@ref)):

- `:double` — ONE filtered Lanczos run with a [`DoubleDeltaFilter`](@ref) (`+1` peak on the lower
  window, `−1` peak on the upper) and a `BothEnds(Window, Window)` target; the pooled result is
  split into the two windows.
- `:separate` — two ordinary single-filter window runs, sequentially. They are sequential by
  design: parallelism already lives inside the matvec, and two Lanczos runs competing for the same
  cores would only slow each other down.

Both paths run in the same units as [`run_polfed_window_pipeline`](@ref): the `Window` criterion is
stated in RESCALED energies, so `mapvals` is the rescaled operator and `eigentol` is passed as
`eigentol / mp.a`; values are un-rescaled once, in `build_polfed_result`.

# Arguments
- `target1`, `target2`: **window** specs (`(lo, hi)`, `(:rescaled, (lo, hi))`, …). They may be
  given in either order; the result is returned in the caller's order. A point target here is an
  error — use the count-based method.

# Keywords
- `strategy::Symbol`: `:auto` (decide from the polynomial orders), `:double`, or `:separate`.
- `mc`, `dosc`, `tol`, `eigentol`, `overestimate_iters`, `layout`: as in [`polfed`](@ref).

# Returns
- [`DoublePolfedResult`](@ref): the two [`PolfedResult`](@ref)s in the caller's argument order,
  plus the strategy decision.
"""
function run_double_polfed_window_pipeline(
    f!,
    x0::AbstractVecOrMat,
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
    spec_a = normalize_target_spec(target1)
    spec_b = normalize_target_spec(target2)
    (spec_a isa TargetWindowSpec && spec_b isa TargetWindowSpec) || throw(ArgumentError(
        "The double window pipeline requires TWO window targets — (:rescaled, (left, right)), " *
        "(:unrescaled, (left, right)), or a bare (left, right). For point targets use the " *
        "count-based `double_polfed(f!, x0, howmany, target1, target2)`.",
    ))

    polfed_log(POLFED_INFO_LEVEL, "double_polfed started (window mode).",
               target1 = format_target_strategy(spec_a), target2 = format_target_strategy(spec_b),
               strategy = strategy, tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    dos = @stage timer :dos calculate_dos(dosc, dos_operator(mp), mp.sys)
    interval_a = @stage timer :interval resolve_window(spec_a, mp.a, mp.b)
    interval_b = @stage timer :interval resolve_window(spec_b, mp.a, mp.b)

    lower, upper, swapped = _double_order_windows(interval_a, interval_b)
    spec_lower, spec_upper = swapped ? (spec_b, spec_a) : (spec_a, spec_b)
    n_lower = howmany_in_interval(dos, lower)
    n_upper = howmany_in_interval(dos, upper)

    strat = @stage timer :filter resolve_double_strategy(lower, upper, strategy)

    mapvals = _make_rescaled(mp.f!, mp.a, mp.b)
    eigentol_rescaled = eigentol / mp.a

    if strat.mode === :double
        # One run, one timer: the two results genuinely describe the same factorization.
        lower_result, upper_result = _double_joint_window(
            mp, strat.double_filter, lower, upper, spec_lower, spec_upper,
            n_lower, n_upper, mapvals, tol, eigentol_rescaled, overestimate_iters, layout, timer,
        )
    else
        # Two runs, two timers seeded with the shared prefix — one shared timer would accumulate
        # both runs' `:lanczos` into a single entry. The first result is detached from its
        # Krylov basis, which is freed before the second run fills its own.
        lower_result = _detached(_double_separate_window_run(
            mp, lower, spec_lower, strat.filter1, n_lower, mapvals, tol, eigentol_rescaled,
            overestimate_iters, layout, _double_run_timer(timer); label = "lower window run",
        ))
        GC.gc(true)
        upper_result = _double_separate_window_run(
            mp, upper, spec_upper, strat.filter2, n_upper, mapvals, tol, eigentol_rescaled,
            overestimate_iters, layout, _double_run_timer(timer); label = "upper window run",
        )
    end

    return _double_result(strat, lower_result, upper_result, swapped)
end
