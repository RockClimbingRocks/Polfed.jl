# DeltaUpFilter leakage validation. The filter has a single `+1` peak; its passband is where the
# polynomial P sits at/above `cutoff`. Over the spectrum [-1, 1] there are three regions:
#   • passband interior            → P amplified   (P ≥ cutoff)
#   • left tail  [-1, passband.left]  → P suppressed  (P < cutoff)
#   • right tail [passband.right, 1]  → P suppressed  (P < cutoff)
# Only the same-sign side lobes (toward the exposed HIGHEST end) count as leaks — the intrinsic
# opposite-sign lobe maps to the far end and is harmless — so the tail scan is signed `direction=+1`
# (leak ⇔ P ≥ cutoff) and the interior `direction=-1` (leak ⇔ P ≤ cutoff), both delegated to the
# shared [`validate_no_leakage_in_interval_numerically`](@ref).
#
# Passband edges sit exactly on `cutoff` by construction, so the interior is shrunk off them by an
# ADAPTIVE margin `min(ε, width/10)` (never emptied). Each tail is scanned coarsely across its whole
# span AND densely over a `2·(passband width)` window right next to the passband, where the first
# (tallest, narrowest) side lobe sits.

"""
    validate_filter(filter::DeltaUpFilter) -> Bool

`true` when `filter` is fit for its requested window. Two families of checks:

1. **Fit for the request** — the passband must **cover** the requested `filter.interval` (nothing
   requested may fall outside the exposed window) and stay **strictly interior** to `(-1, 1)`
   (a passband reaching a spectrum edge means the peak suppresses nothing on that side — the
   near-constant filter a too-wide request degenerates to; edge-filter or ED territory).
2. **No leakage** — the passband interior stays amplified ([`validate_passband`](@ref)) and
   neither tail rises back to `cutoff` ([`validate_tails`](@ref)).

Passband edges land exactly on `cutoff` (the passband is resolved asymmetrically, see
[`resolve_bounds_from_order`](@ref)), so scans are backed off the edges and no overshoot tolerance is
needed. Uses the margin `Δε`, slack `leak_rtol`, and per-region sample counts from
[`DeltaUpFilterDefaults`](@ref).

# Returns
- `true` if the filter fits the request and every region is clean, `false` otherwise.

# Throws
- `ErrorException` if the passband exceeds a spectrum edge ([`validate_passband_within_spectrum`](@ref)).
"""
function validate_filter(filter::DeltaUpSmoothFilter)
    ε         = DeltaUpFilterDefaults.Δε
    leak_rtol = DeltaUpFilterDefaults.leak_rtol
    P         = _scalar_filter(filter)

    validate_passband_within_spectrum(filter)

    # Fit for the request: passband covers the requested interval and stays strictly interior.
    filter.passband.left  <= filter.interval.left  + 1e-8 || return false
    filter.passband.right >= filter.interval.right - 1e-8 || return false
    filter.passband.left > -1.0 + 1e-8 && filter.passband.right < 1.0 - 1e-8 || return false

    npts_pb    = DeltaUpFilterDefaults.npts_within_passband(filter.order)
    npts_tails = DeltaUpFilterDefaults.npts_on_sides_of_the_passband(filter.order)

    passband_ok = validate_passband(P, filter, ε, leak_rtol, npts_pb)
    tails_ok    = validate_tails(P, filter, ε, leak_rtol, npts_tails)
    return passband_ok && tails_ok
end

"""
    validate_passband_within_spectrum(filter::DeltaUpFilter) -> true

Guard that the passband lies inside the spectrum. A single delta filter may legitimately have its
passband reach `±1` (an empty tail is then simply skipped), so this errors only when the passband
goes strictly *outside* `[-1, 1]` — a window that would need a dedicated edge filter.

# Throws
- `ErrorException` if `passband.left < -1` or `passband.right > 1`.
"""
function validate_passband_within_spectrum(filter::DeltaUpFilter)
    filter.passband.left  < -1.0 && error("validate_filter: DeltaUpFilter passband exceeds the left spectrum edge: [$(filter.passband.left), $(filter.passband.right)].")
    filter.passband.right >  1.0 && error("validate_filter: DeltaUpFilter passband exceeds the right spectrum edge: [$(filter.passband.left), $(filter.passband.right)].")
    return true
end

"""
    validate_passband(P, filter::DeltaUpFilter, ε, leak_rtol, npts) -> Bool

Check that `P` stays amplified (`≥ cutoff`) across the passband interior, shrunk off the on-cutoff
edges by the adaptive margin `min(ε, passband_width/10)` so the scan is never empty. Leak ⇔
`P ≤ cutoff` (`direction=-1`).
"""
function validate_passband(P, filter::DeltaUpFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    εp = min(ε, (filter.passband.right - filter.passband.left) / 10)   # adaptive margin, never empties
    lo, hi = filter.passband.left + εp, filter.passband.right - εp
    return validate_no_leakage_in_interval_numerically(P, lo, hi, filter.cutoff, -1; npts = npts, rtol = leak_rtol)
end

"""
    validate_tails(P, filter::DeltaUpFilter, ε, leak_rtol, npts) -> Bool

Check that both tails stay suppressed (`P < cutoff`, leak ⇔ `P ≥ cutoff`, `direction=+1`). Each tail
is scanned twice: **coarsely** across its whole span (`[-1, passband.left-ε]` / `[passband.right+ε, 1]`)
AND **densely** over a `2·(passband width)` window right next to the passband
(`[passband.left-ε-2w, passband.left-ε]` / `[passband.right+ε, passband.right+ε+2w]`), where the first
and tallest side lobe sits. An empty sub-interval (passband reaching a spectrum edge) is skipped.

# Returns
- `true` iff both tails are clean.
"""
function validate_tails(P, filter::DeltaUpFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    cutoff = filter.cutoff
    w  = filter.passband.right - filter.passband.left
    l  = filter.passband.left  - ε
    r  = filter.passband.right + ε

    suppressed(lo, hi) = lo <= hi ?
        validate_no_leakage_in_interval_numerically(P, lo, hi, cutoff, +1; npts = npts, rtol = leak_rtol) : true

    left_ok  = suppressed(-1.0, l) && suppressed(max(-1.0, l - 2w), l)
    right_ok = suppressed(r, 1.0)  && suppressed(r, min(1.0, r + 2w))
    return left_ok && right_ok
end
