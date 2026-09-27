# Shared machinery for the two `double_polfed` pipelines (count and window). Both pipelines have
# the same shape — resolve ONE mapping and ONE DoS, turn the two targets into two ordered
# intervals, decide the strategy, then either run once and split the pool or run twice — and
# differ only in how the intervals are produced and in the units Lanczos works in (a window run is
# stated in RESCALED energies, a count run in true ones). Everything common lives here.

"""
    _double_normalize_howmany(howmany) -> (Int, Int)

Normalize the `howmany` argument of the count-based [`double_polfed`](@ref): an `Integer` means
"that many eigenpairs from EACH window" (not a total to split), a 2-tuple gives the two counts in
the caller's argument order. Both must be positive.
"""
function _double_normalize_howmany(howmany::Integer)
    howmany > 0 || throw(ArgumentError("double_polfed: howmany must be positive; got $howmany."))
    return (Int(howmany), Int(howmany))
end

function _double_normalize_howmany(howmany::Tuple{Integer,Integer})
    (howmany[1] > 0 && howmany[2] > 0) || throw(ArgumentError(
        "double_polfed: both howmany entries must be positive; got $howmany."))
    return (Int(howmany[1]), Int(howmany[2]))
end

_double_normalize_howmany(howmany) = throw(ArgumentError(
    "double_polfed: howmany must be an Integer (the same count for each window) or a " *
    "2-tuple of Integers (one per window); got $(typeof(howmany))."))

"""
    _double_order_windows(interval_a, interval_b) -> (lower, upper, swapped)

Put the two requested windows in spectral order. The double filter's whole convention is
"`interval1` strictly left of `interval2`" ([`DoubleDeltaFilter`](@ref)), so the pipeline works
low→high internally; `swapped` records whether that reversed the caller's argument order, so the
returned [`DoublePolfedResult`](@ref) can be put back into it.

Throws an `ArgumentError` if the windows overlap or merely touch: a single connected region is not
a two-window problem — [`polfed`](@ref) (one filter) or [`patched_polfed`](@ref) (many) covers it,
and a double filter needs a genuine valley between its peaks.
"""
function _double_order_windows(interval_a::Interval, interval_b::Interval)
    swapped = interval_b.left < interval_a.left
    lower, upper = swapped ? (interval_b, interval_a) : (interval_a, interval_b)
    lower.right < upper.left || throw(ArgumentError(
        "double_polfed: the two requested windows overlap or touch — " *
        "[$(lower.left), $(lower.right)] and [$(upper.left), $(upper.right)] (rescaled). " *
        "Two windows must be disjoint with a gap between them; for one connected region use " *
        "`polfed`, or `patched_polfed` if it is wide."))
    return lower, upper, swapped
end

"""
    _double_run_timer(shared::StageTimer) -> StageTimer

A fresh per-run [`StageTimer`](@ref) that already contains the stages the two runs SHARED (the
`:bounds` probe, the `:dos`, the interval/filter resolution). The shared stages are inserted by
reference — they are complete and never written again by the time a run starts — so each run's
report shows the full picture (shared prefix + its own `:transform`/`:lanczos`) without the two
runs' own timings being accumulated into one another.

This is why the `:separate` path does not simply pass the pipeline's timer to both runs:
`StageTimer` stages are **self-accumulating by name**, so one shared timer would report each run's
`:lanczos` as the sum of both.
"""
function _double_run_timer(shared::StageTimer)
    timer = StageTimer()
    for name in shared.order
        timer.children[name] = shared.children[name]
        push!(timer.order, name)
    end
    return timer
end

"""
    _double_lanczos(mp, filter, eig_target, budget, mapvals, tol, eigentol_used,
                    overestimate, layout, timer; label) -> LanczosResult

Build the spectral transformation for `filter` and run one filtered Lanczos factorization — the
single place all four `double_polfed` code paths (count/window × joint/separate) go through.

# Arguments
- `mp::MappingPlan`: the shared mapping (one probe, one rescaling, for both windows).
- `filter`: the resolved filter — a [`DoubleDeltaFilter`](@ref) on the joint path, a single-window
  filter on the separate path.
- `eig_target`: the Lanczos [`EigTarget`](@ref) (a `Window`/count spec, or the `BothEnds` pair the
  joint path builds).
- `budget::Integer`: eigenpair count the Krylov dimension is sized for. On the joint path this is
  `max(n₁, n₂)` — per END, not the total (see [`expected_krylov_dim`](@ref)`(::DoubleDeltaFilter, …)`).
- `mapvals`: the operator the Ritz values are measured against — rescaled for a window run, `mp.f!`
  for a count run.
- `eigentol_used`: eigen-residual tolerance in the SAME units as `mapvals` (so `eigentol / mp.a`
  for a rescaled window run).
- `overestimate::Real`: safety factor inflating the Krylov budget into `maxdim`.

# Returns
- the raw [`LanczosResult`](@ref); post-processing into [`PolfedResult`](@ref)s is the caller's job
  (the joint path builds two results from this one object).
"""
function _double_lanczos(
    mp::MappingPlan,
    filter,
    eig_target,
    budget::Integer,
    mapvals,
    tol::Real,
    eigentol_used::Real,
    overestimate::Real,
    layout::Layout,
    timer::StageTimer;
    label::AbstractString,
)
    n = max(1, Int(budget))
    transformed = @stage timer :transform build_transform(filter, mp)
    expected_dim = expected_krylov_dim(filter, n, mp.sys.block_size, 1.0)
    maxdim = expected_krylov_dim(filter, n, mp.sys.block_size, overestimate)

    log_prelanczos(mp, filter, eig_target, n, expected_dim, maxdim, tol, eigentol_used;
                   label = "double POLFED: $label",
                   rescaled_values = !(mapvals === mp.f!))
    result = @stage timer :lanczos lanczos(transformed, mp.sys.x0, eig_target;
        rot = Defaults.double_polfed.rot,
        basistype = Defaults.double_polfed.basistype,
        maxdim = maxdim,
        tol = tol,
        eigentol = eigentol_used,
        layout = layout,
        mapvals = mapvals,
        next_check = Defaults.double_polfed.conv_checking(expected_dim),
    )
    log_lanczos_outcome(result; label = "double POLFED: $label")
    return result
end

# ── the separate path: one ordinary single-filter run per window ──────────────
# These mirror `run_polfed_window_pipeline` / `run_polfed_target_pipeline` exactly (same targets,
# same units, same budgets) — they only skip the mapping/DoS the double pipeline already did.

"""
    _double_separate_window_run(mp, interval, spec, filter, howmany, mapvals, tol,
                                eigentol_rescaled, overestimate_iters, layout, timer; label)
        -> PolfedResult

One window-target run of the `:separate` path: the same [`Window`](@ref) criterion, rescaled
`mapvals` and rescaled `eigentol` the single-filter window pipeline uses.
"""
function _double_separate_window_run(
    mp::MappingPlan, interval::Interval, spec::TargetSpec, filter, howmany::Integer,
    mapvals, tol::Real, eigentol_rescaled::Real, overestimate_iters::Real,
    layout::Layout, timer::StageTimer; label::AbstractString,
)
    result = _double_lanczos(
        mp, filter, window_eig_target(filter, interval), howmany, mapvals, tol,
        eigentol_rescaled, max(2.5, Float64(overestimate_iters)), layout, timer; label = label,
    )
    return build_polfed_result(result, mp, interval, spec, filter, layout, Int(howmany), timer;
                               vals_rescaled = true)
end

"""
    _double_separate_count_run(mp, interval, spec, filter, howmany, tol, eigentol,
                               overestimate_iters, layout, timer; label) -> PolfedResult

One count-target run of the `:separate` path: the count ([`NumEigs`](@ref)) criterion at the end
the filter exposes, measured against the unrescaled `mp.f!` — the single-filter count pipeline's
convention.

`target_energy` (true energy) is the point the returned pairs are selected nearest to, exactly as
in [`run_polfed_target_pipeline`](@ref); this path runs its own single-peak filter, so its
passband already isolates the window and no bracket limit is needed.
"""
function _double_separate_count_run(
    mp::MappingPlan, interval::Interval, spec::TargetSpec, filter, howmany::Integer,
    tol::Real, eigentol::Real, overestimate_iters::Real,
    layout::Layout, timer::StageTimer; label::AbstractString, target_energy::Float64,
)
    result = _double_lanczos(
        mp, filter, resolve_eig_target(exposed_end(filter), howmany), howmany, mp.f!, tol,
        eigentol, Float64(overestimate_iters), layout, timer; label = label,
    )
    return build_polfed_result(result, mp, interval, spec, filter, layout, Int(howmany), timer;
                               target_energy = target_energy)
end

"""
    _detached(r::PolfedResult) -> PolfedResult

`r` with its eigenvectors copied out of the Krylov basis storage, same `vecs[:, i]` semantics.
The `:separate` path holds the first window's result through the second run, and as a view it
would keep the whole first basis resident next to the second. Follow it with `GC.gc(true)`.
"""
_detached(r::PolfedResult) =
    PolfedResult(r.vals, _own_copy(r.vecs), r.residuals, r.converged, r.layout, r.report)

_own_copy(vecs::Transpose) = transpose(copy(parent(vecs)))   # compact rows, still columns to the caller
_own_copy(vecs::AbstractMatrix) = copy(vecs)

# ── the joint path: one run, two results ──────────────────────────────────────

"""
    _double_bracket_split(mp, lower, upper) -> Float64

The true-energy midpoint of the valley between the two windows — the point past which a window
must NOT adopt a bracket.

The joint run returns both windows in one sorted pool, and window mode widens each window's slice
by one eigenpair on each side to return it bracketed. If no valley eigenpair converged, that
widening would reach across the valley and take the OTHER window's extremal eigenpair. Splitting
the valley at its midpoint makes that impossible (the other window lies entirely beyond it) while
still keeping any genuine valley bracket.
"""
_double_bracket_split(mp::MappingPlan, lower::Interval, upper::Interval) =
    mp.a * (lower.right + upper.left) / 2 + mp.b

"""
    _double_joint_window(mp, filter, lower, upper, spec_lower, spec_upper, n_lower, n_upper,
                         mapvals, tol, eigentol_rescaled, overestimate_iters, layout, timer)
        -> (PolfedResult, PolfedResult)

The joint window run: ONE Lanczos factorization with a `BothEnds(Window, Window)` target
([`double_window_eig_target`](@ref)) whose pooled result is post-processed twice, once per window.

Both results carry the same [`Report`](@ref) — they describe the same run — and each is
bracket-limited to its own side of the valley (see [`_double_bracket_split`](@ref)).
"""
function _double_joint_window(
    mp::MappingPlan, filter::DoubleDeltaFilter, lower::Interval, upper::Interval,
    spec_lower::TargetSpec, spec_upper::TargetSpec, n_lower::Integer, n_upper::Integer,
    mapvals, tol::Real, eigentol_rescaled::Real, overestimate_iters::Real,
    layout::Layout, timer::StageTimer,
)
    result = _double_lanczos(
        mp, filter, double_window_eig_target(filter, lower, upper),
        max(n_lower, n_upper), mapvals, tol, eigentol_rescaled,
        max(2.5, Float64(overestimate_iters)), layout, timer;
        label = "joint window run (both windows)",
    )
    split = _double_bracket_split(mp, lower, upper)
    lower_result = build_polfed_result(
        result, mp, lower, spec_lower, filter, layout, Int(n_lower), timer;
        vals_rescaled = true, bracket_window = (-Inf, split),
    )
    upper_result = build_polfed_result(
        result, mp, upper, spec_upper, filter, layout, Int(n_upper), timer;
        vals_rescaled = true, bracket_window = (split, Inf),
    )
    return lower_result, upper_result
end

"""
    _double_joint_count(mp, filter, lower, upper, spec_lower, spec_upper, n_lower, n_upper,
                        tol, eigentol, overestimate_iters, layout, timer)
        -> (PolfedResult, PolfedResult)

The joint count run: ONE Lanczos factorization with a `BothEnds(NumEigs, NumEigs)` target
([`double_count_eig_target`](@ref) — note it crosses the two counts, since the LOWER window is the
`+1` peak and therefore the transformed HIGH end), post-processed once per window.

Both windows are split out of the same sorted pool, and a count target selects the `n` pairs
NEAREST its target from everything inside the filter's passband. A `DoubleDeltaFilter` has ONE
passband spanning BOTH peaks, so that gate alone cannot separate the windows: a window that
converged fewer than its `n` pairs would reach across the valley and take the other window's
eigenpairs. Each result is therefore bracket-limited to its own side of the valley midpoint,
exactly as the joint WINDOW path is (see [`_double_bracket_split`](@ref)).
"""
function _double_joint_count(
    mp::MappingPlan, filter::DoubleDeltaFilter, lower::Interval, upper::Interval,
    spec_lower::TargetSpec, spec_upper::TargetSpec, n_lower::Integer, n_upper::Integer,
    tol::Real, eigentol::Real, overestimate_iters::Real, layout::Layout, timer::StageTimer;
    target_lower::Float64, target_upper::Float64,
)
    result = _double_lanczos(
        mp, filter, double_count_eig_target(filter, n_lower, n_upper),
        max(n_lower, n_upper), mp.f!, tol, eigentol, Float64(overestimate_iters),
        layout, timer; label = "joint count run (both windows)",
    )
    split = _double_bracket_split(mp, lower, upper)
    lower_result = build_polfed_result(result, mp, lower, spec_lower, filter, layout,
                                       Int(n_lower), timer;
                                       bracket_window = (-Inf, split),
                                       target_energy = target_lower)
    upper_result = build_polfed_result(result, mp, upper, spec_upper, filter, layout,
                                       Int(n_upper), timer;
                                       bracket_window = (split, Inf),
                                       target_energy = target_upper)
    return lower_result, upper_result
end

"""
    _double_result(strategy::DoubleStrategy, lower_result, upper_result, swapped)
        -> DoublePolfedResult

Assemble the returned pair, putting the two results back into the CALLER's argument order
(`swapped` is set when the caller passed the higher window first) and attaching the decision
numbers the strategy was made on.
"""
function _double_result(strategy::DoubleStrategy, lower_result::PolfedResult,
                        upper_result::PolfedResult, swapped::Bool)
    first_result, second_result = swapped ? (upper_result, lower_result) :
                                            (lower_result, upper_result)
    return DoublePolfedResult(
        first_result, second_result, strategy.mode, strategy.reason,
        strategy.order1, strategy.order2, double_order(strategy), strategy.ratio,
    )
end
