# EdgeFilter construction from window geometry. There is nothing to RESOLVE — the order is a
# fixed knob and a monotone filter has no passband to grow — so `build_filter` is construction
# plus placement validation: the single-interval form derives which edge is exposed, the
# two-interval form is the :both seam (mirroring `build_filter(DoubleDeltaFilter, iv1, iv2)`).

# (The "touches a spectrum edge" tolerance `_EDGE_TOL` is shared with `edge_side` and lives in
# filter/edge_side.jl.)

"""
    build_filter(::Type{EdgeFilter}, interval::Interval; order) -> EdgeFilter

Construct the edge filter for a single edge window, deriving `exposed_end` from which spectrum
edge `interval` touches: the left (`interval.left ≤ -1`) exposes `:lowest`, the right
(`interval.right ≥ +1`) exposes `:highest`. Called by [`resolve_filter`](@ref) through dispatch;
nothing further to resolve (fixed order, no passband), so construction is the whole build.

# Throws
- `ArgumentError`: if `interval` touches neither spectrum edge (an interior window — use the
  delta filters) or both (the full spectrum — use exact diagonalization).
"""
function build_filter(::Type{EdgeFilter}, interval::Interval;
                      order::Integer = EdgeFilterDefaults.order)
    touches_left  = interval.left  <= -1.0 + _EDGE_TOL
    touches_right = interval.right >=  1.0 - _EDGE_TOL

    touches_left && touches_right &&
        throw(ArgumentError("EdgeFilter: window [$(interval.left), $(interval.right)] spans the " *
                            "full spectrum; use exact diagonalization instead."))
    touches_left || touches_right ||
        throw(ArgumentError("EdgeFilter: window [$(interval.left), $(interval.right)] touches " *
                            "neither spectrum edge; interior windows take a delta filter."))

    return EdgeFilter(interval, touches_left ? :lowest : :highest; order = order)
end

"""
    build_filter(::Type{EdgeFilter}, interval1::Interval, interval2::Interval; order) -> EdgeFilter

The `:both` form: one filter exposing BOTH spectrum edges (`P(x) = x^order`, `order` odd), for a
[`BothEnds`](@ref) run harvesting `interval1` (at the left edge) and `interval2` (at the right
edge) in a single Krylov space. The stored `interval` is the hull `[interval1.left,
interval2.right]`. Mirrors the two-interval `build_filter(DoubleDeltaFilter, …)` signature — the
seam the future multi-window pipeline resolves through; until that lands (see
`MULTI_WINDOW_TODO.md`) a `:both` filter is used directly with [`build_transform`](@ref), not via
`polfed(...)`.

# Throws
- `ArgumentError`: `interval1` not at the left edge, `interval2` not at the right edge, the
  windows overlapping, or an even `order`.
"""
function build_filter(::Type{EdgeFilter}, interval1::Interval, interval2::Interval;
                      order::Integer = EdgeFilterDefaults.order)
    interval1.left <= -1.0 + _EDGE_TOL ||
        throw(ArgumentError("EdgeFilter (:both): interval1 must touch the left spectrum edge; " *
                            "got [$(interval1.left), $(interval1.right)]."))
    interval2.right >= 1.0 - _EDGE_TOL ||
        throw(ArgumentError("EdgeFilter (:both): interval2 must touch the right spectrum edge; " *
                            "got [$(interval2.left), $(interval2.right)]."))
    interval1.right < interval2.left ||
        throw(ArgumentError("EdgeFilter (:both): the two edge windows must be disjoint; got " *
                            "[$(interval1.left), $(interval1.right)] and [$(interval2.left), $(interval2.right)]."))

    return EdgeFilter(Interval(interval1.left, interval2.right), :both; order = order)
end

# ── resolution interface (see filter.jl: _cutoff_values / _build_candidate) ──────────────────
# The edge filter has no cutoff to ladder — its one resolution parameter is the fixed monomial
# order, so the "ladder" is the single default order. Its `validate_filter` needs no coverage
# condition: a monotone filter exposes the whole ordered spectrum.

"""    _cutoff_values(::Type{EdgeFilter}) — the single candidate: the default monomial order."""
_cutoff_values(::Type{EdgeFilter}) = (EdgeFilterDefaults.order,)

"""    _build_candidate(::Type{EdgeFilter}, interval, order) — the edge filter at monomial `order`."""
_build_candidate(::Type{EdgeFilter}, interval::Interval, order::Real) =
    build_filter(EdgeFilter, interval; order = Int(order))
