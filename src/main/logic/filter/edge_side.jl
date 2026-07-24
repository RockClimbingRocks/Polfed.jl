# Window geometry — the single decision the filter resolution branches on: where does a
# requested interval sit relative to the rescaled spectrum edges ±1? An interval TOUCHING an
# edge is EdgeFilter territory; everything else — including a near-edge window like
# [-0.99, -0.9] — stays delta territory (the cutoff ladder in `resolve_filter` handles the
# boundary leakage a delta peak suffers there). Genuinely cross-filter, so it lives shared.

"""Tolerance for "the bound sits on a spectrum edge" — shared by [`edge_side`](@ref) and the
EdgeFilter build (the pipelines clamp overhanging windows to exactly `±1`, so rounding-scale
slack suffices)."""
const _EDGE_TOL = 1e-9

"""
    _check_interval(interval::Interval) -> Interval

Guard that `interval` is a well-formed rescaled window: finite bounds with
`-1 ≤ left < right ≤ 1`. Returns `interval` so it chains; throws an `ArgumentError` otherwise.
Every `resolve_filter` entry point runs this first — the filters themselves may assume it.
"""
function _check_interval(interval::Interval)
    left, right = interval.left, interval.right
    isfinite(left) && isfinite(right) ||
        throw(ArgumentError("interval bounds must be finite; got [$left, $right]."))
    -1.0 <= left < right <= 1.0 ||
        throw(ArgumentError("interval must satisfy -1 <= left < right <= 1; got [$left, $right]."))
    return interval
end

"""
    edge_side(interval::Interval; atol=_EDGE_TOL) -> Symbol

Which spectrum edge (if any) the rescaled window sits on: `:left` when `interval.left` is within
`atol` of `-1`, `:right` when `interval.right` is within `atol` of `+1`, `:interior` otherwise.
This is the family decision of [`resolve_filter`](@ref): `:left`/`:right` take the monotone
[`EdgeFilter`](@ref); `:interior` takes the delta family — deliberately including near-edge
windows like `[-0.99, -0.9]`, whose boundary leakage the delta cutoff ladder absorbs.

# Throws
- `ArgumentError`: a malformed interval ([`_check_interval`](@ref)), a negative/non-finite
  `atol`, or the full spectrum (both bounds on their edges) — no filter can isolate the whole
  spectrum; that is exact diagonalization.
"""
function edge_side(interval::Interval; atol::Real=_EDGE_TOL)
    _check_interval(interval)
    atol_value = Float64(atol)
    isfinite(atol_value) && atol_value >= 0.0 ||
        throw(ArgumentError("edge tolerance must be finite and non-negative; got $atol."))

    touches_left  = abs(interval.left + 1.0) <= atol_value
    touches_right = abs(interval.right - 1.0) <= atol_value
    touches_left && touches_right &&
        throw(ArgumentError("the window is the full spectrum [-1, 1]; no filter can isolate " *
                            "it — use exact diagonalization instead."))

    touches_left && return :left
    touches_right && return :right
    return :interior
end
