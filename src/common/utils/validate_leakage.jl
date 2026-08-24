"""
    validate_no_leakage_in_interval_numerically(P, lo, hi, cutoff, direction; npts=2000)

Numerically scan the closed interval `[lo, hi]` for same-direction filter leakage.

`P` is the scalar filter polynomial. `cutoff` is the signed cutoff used by the filter, and
`direction` orients the comparison toward the exposed end:

- `direction = +1` for delta-up filters, where leakage means `P(x) >= cutoff`.
- `direction = -1` for delta-down filters, where leakage means `P(x) <= cutoff`.

`rtol` relaxes the threshold to `oriented_cutoff·(1 + rtol)`: since `cutoff` is a soft design
threshold (and the passband boundary sits exactly on it by construction), a side lobe only counts
as a leak once it overshoots the cutoff by more than `rtol`. `rtol = 0` reproduces the strict test.

Returns `true` when none of the sampled points leaks, and `false` on the first sampled leak
(whose location and value are traced at deep-debug level).
The interval is sampled uniformly with `npts` points, including both endpoints.

This is a numerical guard, not a proof: if `npts` is too small, a narrow side lobe between two
sample points can be missed.
"""
function validate_no_leakage_in_interval_numerically(
    P,
    lo::Real,
    hi::Real,
    cutoff::Real,
    direction::Real;
    npts::Int = 2000,
    rtol::Real = 0.0,
)

    npts > 0 || throw(ArgumentError("npts must be positive."))
    lo <= hi || throw(ArgumentError("lo must be <= hi."))
    (direction == 1 || direction == -1) ||
        throw(ArgumentError("direction must be +1 for delta-up or -1 for delta-down."))

    # `cutoff` is a soft design threshold: a side lobe within `rtol` of it (and the passband
    # boundary itself, which sits at the cutoff by construction) is treated as at-cutoff, not a
    # leak. Only overshoots beyond `oriented_cutoff·(1 + rtol)` count.
    oriented_cutoff = direction * cutoff * (1 + rtol)
    for x in range(lo, hi; length = npts)
        if direction * P(x) >= oriented_cutoff
            # WHERE a scan leaks is what tells a filter ladder rung apart from the next; the
            # boolean alone cannot. Emitted here (the only place x is known) at deep-debug level.
            polfed_log(POLFED_DEEP_DEBUG_LEVEL, "Filter leak sampled.",
                       x = x, P = P(x), cutoff = cutoff, direction = direction, scan = (lo, hi))
            return false
        end
    end
    return true
end

