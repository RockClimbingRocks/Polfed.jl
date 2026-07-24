"""
    intervalize_interval(dos, interval, howmany_per_patch) -> Vector{Interval}

Split an interior ownership interval from left to right. Each complete patch contains
approximately `howmany_per_patch` states according to `dos`; the last patch is clipped
to the requested right endpoint. An undersized final remainder is normal and is accepted
without emitting a count-target warning. Edge handling deliberately lives in `_resolve_edges`.
"""
function intervalize_interval(
    dos::DoSResult,
    interval::Interval,
    howmany_per_patch::Integer,
    ;
    edge_distance = nothing,
)
    _check_interval(interval)
    howmany = Int(howmany_per_patch)
    howmany > 0 || throw(ArgumentError(
        "howmany_per_patch must be positive; got $howmany_per_patch.",
    ))

    patches = Interval[]
    left = interval.left
    while left < interval.right
        remaining = integrate_chebyshev_weighted(
            dos.coefficients, left, interval.right,
        ) / π
        if remaining < howmany
            push!(patches, Interval(left, interval.right))
            break
        end

        resolved = resolve_interval(Target(left, howmany, :left), dos)
        right = min(resolved.right, interval.right)
        right > left || error(
            "DoS interval resolution did not advance past $left toward $(interval.right).",
        )
        push!(patches, Interval(left, right))
        left = right
    end
    return patches
end

"""Move `x` outward until the added band contains about `howmany` states."""
function _pad_offset(
    dos::DoSResult,
    x::Float64,
    howmany::Int,
    side::Symbol,
)
    howmany > 0 || return x
    side in (:left, :right) ||
        throw(ArgumentError("padding side must be :left or :right; got :$side."))

    edge = side === :left ? -1.0 : 1.0
    width = side === :left ? x + 1.0 : 1.0 - x
    width > 0 || return edge

    count(p) = side === :left ?
        integrate_chebyshev_weighted(dos.coefficients, x - p, x) / π :
        integrate_chebyshev_weighted(dos.coefficients, x, x + p) / π
    count(width) >= howmany || return edge

    p = bisection(p -> count(p) >= howmany ? -1.0 : 1.0, 0.0, width; tol = 1e-8)
    return side === :left ? x - p : x + p
end

"""Pad one patch by a DoS state count on both sides, clamped to `[-1, 1]`."""
function _pad_interval(dos::DoSResult, interval::Interval, howmany_pad::Int)
    return Interval(
        _pad_offset(dos, interval.left, howmany_pad, :left),
        _pad_offset(dos, interval.right, howmany_pad, :right),
    )
end

"""
    intervalize_interval_overlapping(dos, interval, howmany_per_patch, howmany_pad)

Return aligned `unpadded` ownership patches and `padded` solve patches. The padding is
only overlap; the result buffer reconciles repeated eigenpairs from adjacent solve patches.
"""
function intervalize_interval_overlapping(
    dos::DoSResult,
    interval::Interval,
    howmany_per_patch::Integer,
    howmany_pad::Integer,
    ;
    edge_distance = nothing,
)
    pad = Int(howmany_pad)
    pad >= 0 || throw(ArgumentError("howmany_pad must be non-negative; got $howmany_pad."))
    unpadded = intervalize_interval(dos, interval, howmany_per_patch)
    padded = pad == 0 ? copy(unpadded) :
        [_pad_interval(dos, patch, pad) for patch in unpadded]
    return (; unpadded, padded)
end
