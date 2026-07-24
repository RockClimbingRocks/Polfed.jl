"""
    intervalize_interval(dos, full_iv, howmany_per_patch; edge_distance) -> Vector{Interval}

Partition `full_iv` into contiguous closed intervals. The spectral edge bands
`[-1, -1 + edge_distance]` and `[1 - edge_distance, 1]` are kept as single
pieces because ordinary delta filters should not be placed there. The interior
is divided from left to right using `resolve_interval` so that each complete
interior patch contains approximately `howmany_per_patch` states according to
`dos`. The last interior patch may contain fewer states when it is clipped by
`full_iv` or by the right edge band.

Adjacent returned intervals share their closed endpoint exactly. Downstream
result merging must therefore deduplicate eigenpairs that lie on a shared
boundary.

# Throws
- `ArgumentError` if the requested interval, count, or edge distance is invalid.
- `ErrorException` if DoS-based interval resolution cannot advance to the right.
"""
function intervalize_interval(
    dos::DoSResult,
    full_iv::Interval,
    howmany_per_patch::Integer;
    edge_distance::Real=Defaults.patched_polfed.edge_distance,
)
    howmany = Int(howmany_per_patch)
    howmany > 0 || throw(ArgumentError(
        "intervalize_interval: howmany_per_patch must be positive; got $howmany_per_patch.",
    ))

    distance = Float64(edge_distance)
    isfinite(distance) && 0.0 <= distance < 1.0 || throw(ArgumentError(
        "intervalize_interval: edge_distance must be finite and satisfy " *
        "0 <= edge_distance < 1; got $edge_distance.",
    ))

    left, right = full_iv.left, full_iv.right
    isfinite(left) && isfinite(right) && -1.0 <= left < right <= 1.0 || throw(ArgumentError(
        "intervalize_interval: full_iv must satisfy -1 <= left < right <= 1 " *
        "with finite endpoints; got [$left, $right].",
    ))

    left_delta_edge = -1.0 + distance
    right_delta_edge = 1.0 - distance
    interior_stop = min(right, right_delta_edge)

    function next_right(current::Float64)
        resolved = resolve_interval(Target(current, howmany, :left), dos)
        candidate = min(resolved.right, interior_stop)
        isfinite(candidate) && candidate > current || error(
            "intervalize_interval: DoS interval resolution could not advance past $current " *
            "toward interior endpoint $interior_stop; resolved right=$(resolved.right).",
        )
        return candidate
    end

    first_interval = if right <= left_delta_edge || left >= right_delta_edge
        # The complete request lies inside one edge band.
        full_iv
    elseif left < left_delta_edge
        # Keep the requested part of the left edge band as one interval.
        Interval(left, left_delta_edge)
    else
        # The request starts in the interior delta-filter region.
        Interval(left, next_right(left))
    end

    intervals = Interval[first_interval]
    while last(intervals).right < interior_stop
        current = last(intervals).right
        push!(intervals, Interval(current, next_right(current)))
    end

    if last(intervals).right < right
        push!(intervals, Interval(last(intervals).right, right))
    end
    return intervals
end


"""
    _pad_offset(dos, x, howmany_pad, side) -> Float64

Return the coordinate reached by growing `x` outward — toward `-1` for `side === :left`,
toward `+1` for `side === :right` — until the extension band (`[result, x]` on the left,
`[x, result]` on the right) holds about `howmany_pad` states according to `dos`.

The state count uses the same DoS reconstruction as [`_size_window_from_dos`](@ref):
`(1/π)·`[`integrate_chebyshev_weighted`](@ref) of the kernel-damped Chebyshev coefficients,
integrated exactly. The offset is found by bisecting the band half-width. Growth is clamped
to the rescaled spectrum `[-1, 1]`: if the band all the way to the edge still holds fewer than
`howmany_pad` states (or `x` already sits on the edge), the edge itself (`±1`) is returned.
`howmany_pad == 0` returns `x` unchanged.
"""
function _pad_offset(dos::DoSResult, x::Float64, howmany_pad::Int, side::Symbol)
    howmany_pad > 0 || return x
    edge  = side === :right ? 1.0 : -1.0
    max_p = side === :right ? 1.0 - x : x + 1.0
    max_p > 0.0 || return edge                      # already at (or beyond) the spectrum edge

    coeffs = dos.coefficients
    band(p) = side === :right ?
        integrate_chebyshev_weighted(coeffs, x, x + p) / π :
        integrate_chebyshev_weighted(coeffs, x - p, x) / π

    band(max_p) >= howmany_pad || return edge       # not enough states before the edge: grow to it
    # bfun(0) = +1 (empty band), bfun(max_p) = -1 (holds ≥ howmany_pad): brackets the crossing.
    bfun(p) = band(p) >= howmany_pad ? -1.0 : 1.0
    p = bisection(bfun, 0.0, max_p; tol=1e-8)
    return side === :right ? x + p : x - p
end


"""
    intervalize_interval_overlapping(dos, full_iv, howmany_per_patch, howmany_pad;
                                     edge_distance) -> (; unpadded, padded)

Like [`intervalize_interval`](@ref), but returns **both** the non-overlapping base partition and
an overlapping version of it, so the effect of the padding is explicit at the call site. Every
base patch is grown outward on **both** sides by a band that holds about `howmany_pad` states
(according to `dos`), so adjacent `padded` patches share a genuine overlap region around each
boundary. For example, if the `howmany_pad` band is `0.02` wide, the non-overlapping partition

    unpadded = [-0.1, 0.0], [0.0, 0.1]

becomes

    padded   = [-0.12, 0.02], [-0.02, 0.12]

`unpadded[i]` and `padded[i]` are the same patch before and after padding (equal lengths, aligned
index-for-index). The base partition (its patch positions and count) is produced by
[`intervalize_interval`](@ref) with the same `howmany_per_patch` and `edge_distance`, then each
patch is padded independently. The overlap gives downstream gluing a redundant band on each side
of every patch boundary, so an eigenpair sitting near a boundary is captured by **both**
neighbours (and can be reconciled) rather than depending on a single exact-endpoint hit.

Padding is clamped to the rescaled spectrum `[-1, 1]` (via [`_pad_offset`](@ref)); where a full
`howmany_pad` band does not fit before an edge, that side is grown to the edge. Both outermost
boundaries are padded too, so the `padded` union may extend slightly past `full_iv`.
`howmany_pad == 0` makes `padded` identical to `unpadded`.

A padded edge is never left in the **open edge band** `(1 - edge_distance, 1)` or
`(-1, -1 + edge_distance)`: a value there is too close to `±1` for a delta filter yet not exactly
on the edge, so [`resolve_filter`](@ref) would reject it (neither interior nor edge). Such an edge
is pulled back to the band boundary `±(1 - edge_distance)`, keeping the patch cleanly interior —
the neighbouring edge-band patch (which pads *inward*) already supplies the overlap there. An edge
padded to exactly `±1` is left as is (a genuine [`EdgeFilter`](@ref) patch).

# Arguments
- `dos::DoSResult`: density of states of the **rescaled** operator (coordinates in `[-1, 1]`).
- `full_iv::Interval`: the rescaled region to partition (`-1 <= left < right <= 1`).
- `howmany_per_patch::Integer`: approximate states per base (pre-pad) interior patch.
- `howmany_pad::Integer`: approximate states in each side's overlap band (`>= 0`).

# Returns
A `NamedTuple` `(; unpadded, padded)` of two aligned `Vector{Interval}`s: the base partition and
its outward-padded (overlapping) counterpart.

# Throws
- `ArgumentError` if `howmany_pad < 0`; other argument checks are inherited from
  [`intervalize_interval`](@ref).
"""
function intervalize_interval_overlapping(
    dos::DoSResult,
    full_iv::Interval,
    howmany_per_patch::Integer,
    howmany_pad::Integer;
    edge_distance::Real=Defaults.patched_polfed.edge_distance,
)
    pad = Int(howmany_pad)
    pad >= 0 || throw(ArgumentError(
        "intervalize_interval_overlapping: howmany_pad must be non-negative; got $howmany_pad.",
    ))

    unpadded = intervalize_interval(dos, full_iv, howmany_per_patch; edge_distance = edge_distance)
    pad == 0 && return (; unpadded, padded = copy(unpadded))

    padded = [_pad_interval(dos, iv, pad, edge_distance) for iv in unpadded]
    return (; unpadded, padded)
end


"""
    _pad_interval(dos, iv, pad, edge_distance) -> Interval

Grow one interval `iv` outward by a `pad`-state band on each side ([`_pad_offset`](@ref)), then keep
each padded edge out of the OPEN edge band `(±(1-edge_distance), ±1)`: a value strictly there is
un-filterable (too near `±1` for a delta, not on the edge for an [`EdgeFilter`](@ref)), so it is
pulled back to the band boundary; a value already at `±1` stays (a real edge patch). Used to build
the overlapping partition and to RE-pad wider on a coverage retry.
"""
function _pad_interval(dos::DoSResult, iv::Interval, pad::Int, edge_distance::Real)
    left_band  = -1.0 + Float64(edge_distance)
    right_band =  1.0 - Float64(edge_distance)
    l = _pad_offset(dos, iv.left,  pad, :left)
    r = _pad_offset(dos, iv.right, pad, :right)
    l = (-1.0 < l < left_band) ? left_band  : l
    r = (right_band < r < 1.0) ? right_band : r
    return Interval(l, r)
end



