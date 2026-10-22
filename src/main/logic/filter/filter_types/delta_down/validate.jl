# DeltaDownFilter leakage validation. Mirror of delta_up oriented toward the LOWEST end: the single
# peak is `-1` and `cutoff` is negative, so the passband is where P sits at/below `cutoff`. Over the
# spectrum [-1, 1] there are three regions:
#   • passband interior            → P amplified   (P ≤ cutoff)
#   • left tail  [-1, passband.left]  → P suppressed  (P > cutoff)
#   • right tail [passband.right, 1]  → P suppressed  (P > cutoff)
# The only systematic difference from delta_up is the orientation: the tail scan is `direction=-1`
# (leak ⇔ P ≤ cutoff) and the interior `direction=+1` (leak ⇔ P ≥ cutoff), both delegated to the
# shared [`validate_no_leakage_in_interval_numerically`](@ref).
#
# Passband edges sit exactly on `cutoff` by construction, so the interior is shrunk off them by an
# ADAPTIVE margin `min(ε, width/10)` (never emptied). Each tail is scanned coarsely across its whole
# span AND densely over a `2·(passband width)` window right next to the passband, where the first
# (tallest, narrowest) side lobe sits.

"""
    validate_filter(filter::DeltaDownFilter) -> Bool

`true` when `filter` does not leak toward the lowest end: its passband interior stays amplified
([`validate_passband`](@ref)) and neither tail drops back to the negative `cutoff`
([`validate_tails`](@ref)).

Passband edges land exactly on `cutoff` (resolved asymmetrically, see
[`resolve_bounds_from_order`](@ref)), so scans are backed off the edges and no overshoot tolerance is
needed. Uses the margin `Δε`, slack `leak_rtol`, and per-region sample counts from
[`Defaults.filters.delta_down`](@ref).

# Returns
- `true` if every region is clean, `false` on the first leak.

# Throws
- `ErrorException` if the passband exceeds a spectrum edge ([`validate_passband_within_spectrum`](@ref)).
"""
function validate_filter(filter::DeltaDownFilter)
    ε         = Defaults.filters.delta_down.Δε
    leak_rtol = Defaults.filters.delta_down.leak_rtol
    P         = _scalar_filter(filter)

    validate_passband_within_spectrum(filter)

    # Fit for the request: passband covers the requested interval and stays strictly interior
    # (mirrors DeltaUpFilter — the passband geometry is sign-agnostic).
    filter.passband.left  <= filter.interval.left  + 1e-8 ||
        return _reject(filter, "passband does not cover the request on the left")
    filter.passband.right >= filter.interval.right - 1e-8 ||
        return _reject(filter, "passband does not cover the request on the right")
    filter.passband.left > -1.0 + 1e-8 && filter.passband.right < 1.0 - 1e-8 ||
        return _reject(filter, "passband reaches a spectrum edge — nothing is suppressed there")

    npts_pb    = Defaults.filters.delta_down.npts_within_passband(filter.order)
    npts_tails = Defaults.filters.delta_down.npts_on_sides_of_the_passband(filter.order)

    validate_passband(P, filter, ε, leak_rtol, npts_pb) ||
        return _reject(filter, "the passband interior drops past cutoff — the peak does not cover it")
    validate_tails(P, filter, ε, leak_rtol, npts_tails) ||
        return _reject(filter, "a side lobe in a tail rises back to cutoff (leakage)")
    return true
end

"""
    validate_passband_within_spectrum(filter::DeltaDownFilter) -> true

Guard that the passband lies inside the spectrum. A single delta filter may legitimately have its
passband reach `±1` (an empty tail is then simply skipped), so this errors only when the passband
goes strictly *outside* `[-1, 1]` — a window that would need a dedicated edge filter.

# Throws
- `ErrorException` if `passband.left < -1` or `passband.right > 1`.
"""
function validate_passband_within_spectrum(filter::DeltaDownFilter)
    filter.passband.left  < -1.0 && error("validate_filter: DeltaDownFilter passband exceeds the left spectrum edge: [$(filter.passband.left), $(filter.passband.right)].")
    filter.passband.right >  1.0 && error("validate_filter: DeltaDownFilter passband exceeds the right spectrum edge: [$(filter.passband.left), $(filter.passband.right)].")
    return true
end

"""
    validate_passband(P, filter::DeltaDownFilter, ε, leak_rtol, npts) -> Bool

Check that `P` stays amplified (`≤ cutoff`, the negative threshold) across the passband interior,
shrunk off the on-cutoff edges by the adaptive margin `min(ε, passband_width/10)` so the scan is
never empty. Leak ⇔ `P ≥ cutoff` (`direction=+1`).
"""
function validate_passband(P, filter::DeltaDownFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    εp = min(ε, (filter.passband.right - filter.passband.left) / 10)   # adaptive margin, never empties
    lo, hi = filter.passband.left + εp, filter.passband.right - εp
    return validate_no_leakage_in_interval_numerically(P, lo, hi, filter.cutoff, +1; npts = npts, rtol = leak_rtol)
end

"""
    validate_tails(P, filter::DeltaDownFilter, ε, leak_rtol, npts) -> Bool

Check that both tails stay suppressed (`P > cutoff`, leak ⇔ `P ≤ cutoff`, `direction=-1`). Each tail
is scanned twice: **coarsely** across its whole span (`[-1, passband.left-ε]` / `[passband.right+ε, 1]`)
AND **densely** over a `2·(passband width)` window right next to the passband
(`[passband.left-ε-2w, passband.left-ε]` / `[passband.right+ε, passband.right+ε+2w]`), where the first
and tallest side lobe sits. An empty sub-interval (passband reaching a spectrum edge) is skipped.

# Returns
- `true` iff both tails are clean.
"""
function validate_tails(P, filter::DeltaDownFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    cutoff = filter.cutoff
    w  = filter.passband.right - filter.passband.left
    l  = filter.passband.left  - ε
    r  = filter.passband.right + ε

    suppressed(lo, hi) = lo <= hi ?
        validate_no_leakage_in_interval_numerically(P, lo, hi, cutoff, -1; npts = npts, rtol = leak_rtol) : true

    left_ok  = suppressed(-1.0, l) && suppressed(max(-1.0, l - 2w), l)
    right_ok = suppressed(r, 1.0)  && suppressed(r, min(1.0, r + 2w))
    return left_ok && right_ok
end
