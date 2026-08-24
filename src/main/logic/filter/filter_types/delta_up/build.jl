# DeltaUpFilter construction: how a window becomes a resolved filter. The order is bisected so the
# polynomial drops to `cutoff` at the requested window's edges (then shrunk by the safety factor,
# which widens the passband); the effective passband is recovered from that order.
#
# The delta filter is symmetric (target ± δ) and peaks at +1, so both the order predicate and the
# passband growth test the polynomial directly (no peak-sign abstraction needed). The reusable
# bracket-and-bisect mechanic lives in the shared `bisect_order` / `bisection`.

"""
    analytical_order(filter::DeltaUpFilter, δ::Real) -> Float64

Heuristic Chebyshev order for a window of half-width `δ` around `λ = filter.target`:
`K ≈ c · √(1 - λ²) / δ` with `c = `[`Defaults.filters.delta_up.order_bracket_constant`](@ref).
Delta-filter-specific; used only to *bracket* the exact [`bisect_order`](@ref).
"""
analytical_order(filter::DeltaUpFilter, δ::Real) =
    Defaults.filters.delta_up.order_bracket_constant * sqrt(1 - filter.target^2) / δ

"""
    resolve_order!(filter::DeltaUpFilter) -> Int

Resolve (and store) the polynomial order for `filter.interval`: the smallest order at which the
normalized polynomial has fallen below `cutoff` at **both** window endpoints, then shrunk by
`order_safety_factor` (`< 1` ⇒ a wider passband so every requested eigenpair sits comfortably
inside). Uses the shared [`bisect_order`](@ref) with a delta-specific "suppressed" predicate.

Mutates `filter.order` (the struct is mutable and built once); returns the resolved order.
"""
function resolve_order!(filter::DeltaUpFilter)
    l, r = filter.interval.left, filter.interval.right

    # +1 peak: "suppressed" ⇔ P has dropped below cutoff at both edges. (Trial orders mutate
    # filter.order in place — cheap, and build_filter overwrites it with the final value.)
    suppressed(K::Int) = begin
        filter.order = K
        P = _scalar_filter(filter)
        min(P(l), P(r)) < filter.cutoff
    end

    K = bisect_order(analytical_order(filter, (r - l) / 2), suppressed)
    filter.order = max(1, floor(Int, K * filter.order_safety_factor))
end

"""
    resolve_bounds_from_order(filter::DeltaUpFilter) -> Interval

The filter's **effective passband** at its resolved order: the window `[left, right]` (clamped to
`[-1, 1]`) where the polynomial sits at or above `cutoff`. The half-width is grown **independently
on each side** to the nearest cutoff crossing, so the passband is generally **asymmetric** — `P` is
symmetric about `0`, not about a non-central `target`, so the two sides drop off at different rates.
(A symmetric `target ± δ` would leave the slower-dropping edge above cutoff, which
[`validate_filter`](@ref) would then flag as spurious boundary leakage.)
"""
function resolve_bounds_from_order!(filter::DeltaUpFilter)
    P = _scalar_filter(filter)
    c = filter.cutoff
    t = filter.target

    # +1 peak: "inside" the passband ⇔ P ≥ cutoff. Walk away from the target and take the
    # nearest crossing. The walk is split at the requested interval edge so the bisection bracket
    # is local to the contract we care about:
    #   target -> interval edge -> spectrum edge
    # If a fixed/unsafe order already puts the interval edge outside, the first segment returns the
    # inner crossing; otherwise the second segment grows the passband outward.
    g(x) = P(x) - c
    θstep = π / max(4096, 64 * max(filter.order, 1))
    crossing_atol = 1e-10   # guaranteed value accuracy of the crossing: |P(edge) − cutoff| ≤ this
    root_tol = 1e-13        # a scan sample this close to the root is already on it — take it directly

    function first_crossing(x0, x1)
        x0 == x1 && return nothing

        θ0 = acos(clamp(x0, -1.0, 1.0))
        θ1 = acos(clamp(x1, -1.0, 1.0))
        nsteps = max(1, ceil(Int, abs(θ1 - θ0) / θstep))

        previous_x = x0
        previous_θ = θ0
        previous_g = g(previous_x)
        abs(previous_g) <= root_tol && return previous_x

        for i in 1:nsteps
            θ = θ0 + (θ1 - θ0) * (i / nsteps)
            x = cos(θ)
            gx = g(x)

            abs(gx) <= root_tol && return x
            if previous_g * gx < 0
                θroot = refine_crossing(θ -> g(cos(θ)), min(previous_θ, θ), max(previous_θ, θ);
                                        atol = crossing_atol)
                return cos(θroot)
            end

            previous_x = x
            previous_θ = θ
            previous_g = gx
        end

        return nothing
    end

    function edge(sgn)
        endpoint = sgn < 0 ? -1.0 : 1.0
        interval_edge = sgn < 0 ? filter.interval.left : filter.interval.right

        crossing = first_crossing(t, interval_edge)
        crossing !== nothing && return crossing

        crossing = first_crossing(interval_edge, endpoint)
        crossing !== nothing && return crossing

        return endpoint
    end

    filter.passband = Interval(edge(-1.0), edge(+1.0))
end

"""
    build_filter(::Type{DeltaUpFilter}, interval::Interval; cutoff, order, order_safety_factor)

Construct the resolved filter for a window: build from `interval` (target = midpoint), resolve the
order from `cutoff` unless a fixed `order ≥ 0` was given, then compute and store the effective
`passband`. Called by [`resolve_filter`](@ref) through dispatch.
"""
function build_filter(::Type{DeltaUpFilter}, interval::Interval;
                      cutoff::Real              = Defaults.filters.delta_up.cutoff,
                      order::Integer            = Defaults.filters.delta_up.order,
                      order_safety_factor::Real = Defaults.filters.delta_up.order_safety_factor)
                      
    filter = DeltaUpFilter(interval; cutoff = cutoff, order = order,
                           order_safety_factor = order_safety_factor)
    filter.order < 0 && resolve_order!(filter)    # sentinel ⇒ resolve; else keep user order
    resolve_bounds_from_order!(filter)
    return filter
end

# ── resolution interface (see filter.jl: _cutoff_values / _build_candidate) ──────────────────

"""    _cutoff_values(::Type{DeltaUpFilter}) — the cutoff ladder `0.17…0.35`, sharpest first."""
_cutoff_values(::Type{DeltaUpFilter}) = Defaults.filters.delta_up.cutoff_ladder

"""    _build_candidate(::Type{DeltaUpFilter}, interval, cutoff) — one fresh filter at `cutoff`."""
_build_candidate(::Type{DeltaUpFilter}, interval::Interval, cutoff::Real) =
    build_filter(DeltaUpFilter, interval; cutoff = cutoff)
