# DeltaDownFilter construction: identical to delta_up's, but every cutoff test is oriented toward
# the LOWEST end — the peak is -1 and `cutoff` is negative, so the polynomial is compared directly
# as `P(x) ≤ cutoff`. This is the single systematic difference from delta_up/build.jl.

"""    analytical_order(filter::DeltaDownFilter, δ) -> Float64
`K ≈ c·√(1-λ²)/δ` with `c = `[`Defaults.filters.delta_down.order_bracket_constant`](@ref) —
the delta-filter order bracket (same value as the up filter)."""
analytical_order(filter::DeltaDownFilter, δ::Real) =
    Defaults.filters.delta_down.order_bracket_constant * sqrt(1 - filter.target^2) / δ


"""
    resolve_order!(filter::DeltaDownFilter) -> Int

Resolve (and store) the polynomial order for `filter.interval`: the first order at which either
requested endpoint has risen above the negative `cutoff`, then shrunk by `order_safety_factor`
(`< 1` ⇒ a wider passband so every requested eigenpair stays inside). Uses the shared
[`bisect_order`](@ref) with a delta-down-specific "suppressed" predicate.

Mutates `filter.order` (the struct is mutable and built once); returns the resolved order.
"""
function resolve_order!(filter::DeltaDownFilter)
    l, r = filter.interval.left, filter.interval.right

    suppressed(K::Int) = begin
        filter.order = K
        P = _scalar_filter(filter)
        max(P(l), P(r)) > filter.cutoff  # -1 peak: cutoff is negative
    end

    K = bisect_order(analytical_order(filter, (r - l) / 2), suppressed)
    filter.order = max(1, floor(Int, K * filter.order_safety_factor))
    return filter.order
end

"""
    resolve_bounds_from_order(filter::DeltaDownFilter) -> Interval

The filter's **effective passband** at its resolved order: the window `[left, right]` (clamped to
`[-1, 1]`) where the polynomial sits at or below the negative `cutoff`. The half-width is grown
independently on each side to the nearest cutoff crossing, so the passband is generally
**asymmetric** — `P` is symmetric about `0`, not about a non-central `target`.
"""
function resolve_bounds_from_order!(filter::DeltaDownFilter)
    P = _scalar_filter(filter)
    c = filter.cutoff
    t = filter.target

    # -1 peak: "inside" the passband ⇔ P ≤ cutoff. Walk away from the target and take the nearest
    # crossing. The walk is split at the requested interval edge so the bisection bracket is local
    # to the contract we care about:
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
    build_filter(::Type{DeltaDownFilter}, interval; cutoff, order, order_safety_factor)

    Construct + resolve the filter for a window (target = midpoint, resolve order unless fixed, then
    store the effective `passband`). Called by [`resolve_filter`](@ref) through dispatch.
"""
function build_filter(::Type{DeltaDownFilter}, interval::Interval;
                      cutoff::Real              = Defaults.filters.delta_down.cutoff,
                      order::Integer            = Defaults.filters.delta_down.order,
                      order_safety_factor::Real = Defaults.filters.delta_down.order_safety_factor)
                      
    filter = DeltaDownFilter(interval; cutoff = cutoff, order = order,
                             order_safety_factor = order_safety_factor)
    filter.order < 0 && resolve_order!(filter)
    resolve_bounds_from_order!(filter)
    return filter
end

# ── resolution interface (see filter.jl: _cutoff_values / _build_candidate) ──────────────────

"""    _cutoff_values(::Type{DeltaDownFilter}) — the negated cutoff ladder `-0.17…-0.35`."""
_cutoff_values(::Type{DeltaDownFilter}) = Defaults.filters.delta_down.cutoff_ladder

"""    _build_candidate(::Type{DeltaDownFilter}, interval, cutoff) — one fresh filter at `cutoff`."""
_build_candidate(::Type{DeltaDownFilter}, interval::Interval, cutoff::Real) =
    build_filter(DeltaDownFilter, interval; cutoff = cutoff)
