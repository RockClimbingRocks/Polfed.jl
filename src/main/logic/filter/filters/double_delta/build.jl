# Order & passband resolution for DoubleDeltaFilter.
#
# The double filter's two peaks (target1 → +1, target2 → −1) CROSS-TALK: adding one lobe changes the
# other lobe's values at every endpoint, and the 2×2 mixing solve in `coefficients` re-weights both
# whenever either order changes. So the order MUST be resolved on the COMBINED polynomial
# `P = _scalar_filter(filter)` — resolving each lobe in isolation lets the effective passband drift
# off the requested window. Because the two orders are coupled, they are resolved by coordinate
# descent (alternate order1/order2 until neither moves), then a coverage-repair pass GUARANTEES both
# windows sit inside their passbands.

# ── order resolution on the COMBINED polynomial ────────────────────────────────

_double_halfwidth(iv::Interval) = (iv.right - iv.left) / 2

"""Heuristic Chebyshev order for a delta peak of half-width `δ` at `target` (delta-filter bracket)."""
_double_analytical_order(target::Real, δ::Real) = 2.65499 * sqrt(1 - target^2) / δ

"""
    _resolve_order1!(filter, order2_fixed) -> Int

Resolve `order1` (with `order2` held at `order2_fixed`) on the COMBINED polynomial. The binding edge
of window 1 is the LOWER of its two endpoints — `min(P(l), P(r))` — because the inner edge is pulled
DOWN by the `−1` peak's cross-talk. [`bisect_order`](@ref) finds the smallest order at which that
binding edge first drops BELOW `cutoff_upper` (the window becomes uncovered as the peak sharpens);
`order_safety_factor < 1` then pulls back to a covered, slightly wider peak. Mutates & returns `order1`.
"""
function _resolve_order1!(filter::DoubleDeltaFilter, order2_fixed::Int)
    l, r = filter.interval1.left, filter.interval1.right
    uncovered(K::Int) = begin
        filter.order1, filter.order2 = K, order2_fixed
        P = _scalar_filter(filter)
        min(P(l), P(r)) < filter.cutoff_upper
    end
    K = bisect_order(_double_analytical_order(filter.target1, _double_halfwidth(filter.interval1)), uncovered)
    filter.order1 = max(1, floor(Int, K * filter.order_safety_factor))
    return filter.order1
end

"""
    _resolve_order2!(filter, order1_fixed) -> Int

Mirror of [`_resolve_order1!`](@ref) for the `−1` peak: the binding edge of window 2 is the HIGHER of
its endpoints — `max(P(l), P(r))` — pulled UP by the `+1` peak. [`bisect_order`](@ref) finds the
smallest order at which it first rises ABOVE `cutoff_lower`, then `order_safety_factor` widens it.
Mutates & returns `order2`.
"""
function _resolve_order2!(filter::DoubleDeltaFilter, order1_fixed::Int)
    l, r = filter.interval2.left, filter.interval2.right
    uncovered(K::Int) = begin
        filter.order1, filter.order2 = order1_fixed, K
        P = _scalar_filter(filter)
        max(P(l), P(r)) > filter.cutoff_lower
    end
    K = bisect_order(_double_analytical_order(filter.target2, _double_halfwidth(filter.interval2)), uncovered)
    filter.order2 = max(1, floor(Int, K * filter.order_safety_factor))
    return filter.order2
end

"""
    resolve_order!(filter::DoubleDeltaFilter; max_passes=8) -> DoubleDeltaFilter

Resolve BOTH orders on the combined polynomial by coordinate descent: seed each with its analytical
single-lobe estimate (so the first combined `P` is well-formed), then alternate [`_resolve_order1!`](@ref)
/ [`_resolve_order2!`](@ref) until neither order changes (or `max_passes` is hit). Coordinate descent
handles the coupling between the two peak widths; the follow-up [`_repair_coverage!`](@ref) guarantees
the final coverage. Mutates `order1`/`order2`; returns `filter`.
"""
function resolve_order!(filter::DoubleDeltaFilter; max_passes::Int = 8)
    filter.order1 = max(1, ceil(Int, _double_analytical_order(filter.target1, _double_halfwidth(filter.interval1))))
    filter.order2 = max(1, ceil(Int, _double_analytical_order(filter.target2, _double_halfwidth(filter.interval2))))
    prev1, prev2 = filter.order1, filter.order2
    for _ in 1:max_passes
        _resolve_order1!(filter, filter.order2)
        _resolve_order2!(filter, filter.order1)
        (filter.order1 == prev1 && filter.order2 == prev2) && break
        prev1, prev2 = filter.order1, filter.order2
    end
    return filter
end

# ── coverage guarantee ─────────────────────────────────────────────────────────

"""    _covered(filter) -> (up_ok, dn_ok) — is each window fully on its peak's side of the cutoff (combined P)?"""
function _covered(filter::DoubleDeltaFilter)
    P  = _scalar_filter(filter)
    up = P(filter.interval1.left) >= filter.cutoff_upper && P(filter.interval1.right) >= filter.cutoff_upper
    dn = P(filter.interval2.left) <= filter.cutoff_lower && P(filter.interval2.right) <= filter.cutoff_lower
    return up, dn
end

"""
    _repair_coverage!(filter; max_iters=1000) -> DoubleDeltaFilter

Guarantee both windows sit inside their passbands: while a window is not yet covered, WIDEN
(decrement) that peak's order. Widening a peak always helps its own window's coverage, and the
fully-wide limit covers both, so this terminates — landing on the sharpest orders that still cover.
"""
function _repair_coverage!(filter::DoubleDeltaFilter; max_iters::Int = 1000)
    for _ in 1:max_iters
        up, dn = _covered(filter)
        (up && dn) && break
        !up && filter.order1 > 1 && (filter.order1 -= 1)
        !dn && filter.order2 > 1 && (filter.order2 -= 1)
        (filter.order1 == 1 && filter.order2 == 1) && break   # can't widen further (windows too close)
    end
    return filter
end

# ── passband resolution ───────────────────────────────────────────────────────
# The double filter has TWO peaks (target1 → +1, target2 → −1), hence TWO passbands. Each is found
# exactly like a single delta filter's — grow outward from the peak until the COMBINED polynomial P
# crosses that peak's cutoff — just done twice. `_peak_passband` is the shared single-peak scanner;
# `resolve_bounds_from_order!` runs it for each peak and stores both passbands.

"""
    _peak_passband(P, target, cutoff, interval, order) -> Interval

The passband around ONE peak: the window `[left, right] ⊆ [-1, 1]` where the (combined) polynomial
`P` stays on the peak's side of `cutoff`. Each edge is the nearest point — walking outward from
`target` — where `P` crosses `cutoff`.

The crossing is located in `θ = acos(x)` space (where the Chebyshev oscillations are uniform):
scan in fine steps for a sign change of `g(x) = P(x) − cutoff`, then refine it by bisection — so it
robustly returns the FIRST crossing even when `P` wiggles (e.g. the edge sitting in the valley
between the two peaks). The walk is split `target → interval edge → spectrum edge`: if the requested
`interval` already reaches past the crossing the inner segment finds it, otherwise the outer segment
grows the passband outward; if `P` never crosses, that spectrum edge is returned. Sign-agnostic —
works for the `+1` peak (P dropping to a positive cutoff) and the `−1` peak (P rising to a negative
cutoff) alike.
"""
function _peak_passband(P, target::Real, cutoff::Real, interval::Interval, order::Int)
    g(x) = P(x) - cutoff
    θstep    = π / max(4096, 64 * max(order, 1))   # scan resolution in acos-space (finer for higher order)
    crossing_atol = 1e-10   # guaranteed value accuracy of the crossing: |P(edge) − cutoff| ≤ this
    root_tol      = 1e-13   # a scan sample this close to the root is already on it — take it directly

    # First x on [x0, x1] (walking in θ) where P crosses cutoff, or `nothing` if it never does.
    function first_crossing(x0, x1)
        x0 == x1 && return nothing
        θ0, θ1 = acos(clamp(x0, -1.0, 1.0)), acos(clamp(x1, -1.0, 1.0))
        nsteps = max(1, ceil(Int, abs(θ1 - θ0) / θstep))

        prev_x, prev_θ = x0, θ0
        prev_g = g(prev_x)
        abs(prev_g) <= root_tol && return prev_x

        for i in 1:nsteps
            θ  = θ0 + (θ1 - θ0) * (i / nsteps)
            x  = cos(θ)
            gx = g(x)
            abs(gx) <= root_tol && return x
            if prev_g * gx < 0                                    # sign change ⇒ crossing in (prev, current)
                θroot = refine_crossing(θ -> g(cos(θ)), min(prev_θ, θ), max(prev_θ, θ); atol = crossing_atol)
                return cos(θroot)
            end
            prev_x, prev_θ, prev_g = x, θ, gx
        end
        return nothing
    end

    # One passband edge: walk from the target toward the spectrum edge (sgn = -1 left / +1 right).
    function edge(sgn)
        spectrum_edge = sgn < 0 ? -1.0 : 1.0
        interval_edge = sgn < 0 ? interval.left : interval.right
        crossing = first_crossing(target, interval_edge)          # crossing inside the requested window?
        crossing !== nothing && return crossing
        crossing = first_crossing(interval_edge, spectrum_edge)   # else grow outward to the crossing
        crossing !== nothing && return crossing
        return spectrum_edge                                      # P never crosses ⇒ passband hits the edge
    end

    return Interval(edge(-1.0), edge(+1.0))
end

"""
    resolve_bounds_from_order!(filter::DoubleDeltaFilter) -> DoubleDeltaFilter

Resolve and store BOTH effective passbands from the combined polynomial `P = _scalar_filter(filter)`:
`passband1` around the `+1` peak (`target1`, `cutoff_upper`) and `passband2` around the `−1` peak
(`target2`, `cutoff_lower`), each via [`_peak_passband`](@ref). Used by `validate_filter`.
"""
function resolve_bounds_from_order!(filter::DoubleDeltaFilter)
    P = _scalar_filter(filter)
    filter.passband1 = _peak_passband(P, filter.target1, filter.cutoff_upper, filter.interval1, filter.order1)
    filter.passband2 = _peak_passband(P, filter.target2, filter.cutoff_lower, filter.interval2, filter.order2)
    return filter
end

# ── top-level builder ──────────────────────────────────────────────────────────

"""
    build_filter(::Type{DoubleDeltaFilter}, interval1, interval2; kwargs...) -> DoubleDeltaFilter

Construct the resolved double-delta filter for two windows (`interval1` strictly left of `interval2`,
targets = midpoints, mapping to `+1`/`−1`). When either order is left at the `-1` sentinel, resolve
BOTH orders on the combined polynomial by coordinate descent ([`resolve_order!`](@ref)) and guarantee
both windows sit inside their passbands ([`_repair_coverage!`](@ref)); user-fixed orders are kept as
given. Then compute and store both effective passbands.

# Keywords
- `order1`, `order2::Integer = -1`: fixed orders, or `-1` to resolve.
- `cutoff_lower`, `cutoff_upper::Real`: the `−1`-peak / `+1`-peak cutoffs.
- `order_safety_factor::Real`: `< 1` widens the resolved peaks (safety margin).
"""
function build_filter(::Type{DoubleDeltaFilter}, interval1::Interval, interval2::Interval;
                    order1::Integer             = DoubleDeltaFilterDefaults.order,
                    order2::Integer             = DoubleDeltaFilterDefaults.order,
                    cutoff_lower::Real          = DoubleDeltaFilterDefaults.cutoff_lower,
                    cutoff_upper::Real          = DoubleDeltaFilterDefaults.cutoff_upper,
                    order_safety_factor::Real   = DoubleDeltaFilterDefaults.order_safety_factor)

    filter = DoubleDeltaFilter(interval1, interval2;
        cutoff_lower = cutoff_lower,
        cutoff_upper = cutoff_upper,
        order1 = order1,
        order2 = order2,
        order_safety_factor = order_safety_factor,
    )

    if filter.order1 < 0 || filter.order2 < 0
        resolve_order!(filter)       # coordinate descent on the combined polynomial
        _repair_coverage!(filter)    # guarantee both windows ⊆ their passbands
    end
    resolve_bounds_from_order!(filter)

    return filter
end

# ── resolution interface (see filter.jl: _cutoff_values / _build_candidate) ──────────────────
# The double filter's "request" is the two-interval Tuple (iv1, iv2) — one ladder value sets
# both cutoffs symmetrically. The per-window coverage check lives in `validate_filter` (the
# filter carries both requested intervals).

"""    _cutoff_values(::Type{DoubleDeltaFilter}) — cutoff magnitudes `0.17…0.35` (used as `∓c`)."""
_cutoff_values(::Type{DoubleDeltaFilter}) = DoubleDeltaFilterDefaults.cutoff_ladder

"""    _build_candidate(::Type{DoubleDeltaFilter}, (iv1, iv2), c) — one fresh filter at `∓c`."""
_build_candidate(::Type{DoubleDeltaFilter}, intervals::Tuple{Interval, Interval}, cutoff::Real) =
    build_filter(DoubleDeltaFilter, intervals[1], intervals[2];
                 cutoff_lower = -cutoff, cutoff_upper = cutoff)
