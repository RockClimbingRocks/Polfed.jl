# Numerical leakage validation for the DoubleDeltaFilter. The combined polynomial P must, over the
# whole spectrum [-1, 1], behave in five regions:
#   • passband1 interior  → P amplified above cutoff_upper (the +1 peak)
#   • passband2 interior  → P suppressed below cutoff_lower (the −1 peak)
#   • left outer tail  / valley / right outer tail → P confined to the band [cutoff_lower, cutoff_upper]
# Passband edges sit exactly on the cutoffs by construction, so every scanned region is shrunk/grown
# off those edges to avoid re-flagging them. The shrink margin is ADAPTIVE — `min(Δε, width/10)` — so
# a very narrow passband or valley is still scanned (collapses toward its center) rather than emptied
# and skipped.

"""
    _within_band(P, lo, hi, lower, upper; npts, rtol) -> Bool

Test whether the scalar polynomial `P` stays inside the band `[lower, upper]` across `[lo, hi]`.

This is the two-sided suppression test used in the tail and valley regions, composed from the shared
one-sided [`validate_no_leakage_in_interval_numerically`](@ref) run on each side — the ONE place the
`direction` convention is spelled out:
- `(upper, +1)`: a leak is `P(x) ≥ upper`, so this enforces `P ≤ upper`.
- `(lower, -1)`: a leak is `P(x) ≤ lower`, so this enforces `P ≥ lower`.

# Arguments
- `P`: the combined filter polynomial (callable, e.g. from [`_scalar_filter`](@ref)).
- `lo`, `hi`: closed scan interval (`lo ≤ hi`).
- `lower`, `upper`: the band bounds (`lower ≤ upper`).

# Keyword Arguments
- `npts::Int`: number of uniform samples across `[lo, hi]`.
- `rtol::Real`: relative slack on each bound (see the one-sided test); `0` is the strict check.

# Returns
- `true` if no sample escapes the band, `false` on the first sample that does.

# Notes
A sampling guard, not a proof: a narrow lobe between two samples can be missed if `npts` is too low.
"""
function _within_band(P, lo::Real, hi::Real, lower::Real, upper::Real; npts::Int, rtol::Real)
    below_upper = validate_no_leakage_in_interval_numerically(P, lo, hi, upper, +1; npts = npts, rtol = rtol)
    above_lower = validate_no_leakage_in_interval_numerically(P, lo, hi, lower, -1; npts = npts, rtol = rtol)
    return below_upper && above_lower
end

"""
    validate_filter(filter::DoubleDeltaFilter) -> Bool

Numerically validate that `filter`'s combined polynomial `P` does not leak anywhere on `[-1, 1]`.

Guards first that both passbands lie inside the spectrum ([`validate_passbands_within_spectrum`](@ref))
and that **each passband covers its requested window** (`passband1 ⊇ interval1`,
`passband2 ⊇ interval2` — the fit-for-request check), then checks the five regions with
per-region sample counts from `DoubleDeltaFilterDefaults`:
the two passband interiors are amplified past their cutoffs ([`validate_passbands`](@ref)), and the
two outer tails ([`validate_outer_tails`](@ref)) and the valley ([`validate_between_passbands`](@ref))
stay inside the band `[cutoff_lower, cutoff_upper]`.

# Returns
- `true` if every region is clean, `false` if any region leaks.

# Throws
- `ErrorException` if a passband reaches past a spectrum edge (a near-edge window needs a dedicated
  edge filter, not this one).

# Notes
Uses the filter's own `Δε` / `leak_rtol` defaults and order-dependent sample counts; a numerical
guard, not a proof (see [`_within_band`](@ref)).
"""
function validate_filter(filter::DoubleDeltaFilter)
    ε         = DoubleDeltaFilterDefaults.Δε
    leak_rtol = DoubleDeltaFilterDefaults.leak_rtol
    P         = _scalar_filter(filter)

    validate_passbands_within_spectrum(filter, ε)   # guard: tails must lie inside [-1, 1] (else edge filter)

    # Fit for the request: EACH passband must cover its requested window.
    filter.passband1.left  <= filter.interval1.left  + 1e-8 || return false
    filter.passband1.right >= filter.interval1.right - 1e-8 || return false
    filter.passband2.left  <= filter.interval2.left  + 1e-8 || return false
    filter.passband2.right >= filter.interval2.right - 1e-8 || return false

    npts_pb      = DoubleDeltaFilterDefaults.npts_within_passband(filter.order1, filter.order2)
    npts_tails   = DoubleDeltaFilterDefaults.npts_on_sides_of_the_passband(filter.order1, filter.order2)
    npts_valley  = DoubleDeltaFilterDefaults.npts_between_passbands(filter.order1, filter.order2)

    passbands_ok = validate_passbands(P, filter, ε, leak_rtol, npts_pb)
    tails_ok     = validate_outer_tails(P, filter, ε, leak_rtol, npts_tails)
    valley_ok    = validate_between_passbands(P, filter, ε, leak_rtol, npts_valley)

    return passbands_ok && tails_ok && valley_ok
end

"""
    validate_passbands_within_spectrum(filter::DoubleDeltaFilter, ε) -> true

Guard that neither passband reaches past a spectrum edge, so the outer-tail scan intervals
(`[-1, passband1.left-ε]` and `[passband2.right+ε, 1]`) stay well-formed.

# Returns
- `true` when both passbands lie inside `[-1, 1]` (with the `ε` margin).

# Throws
- `ErrorException` with a clear message if `passband1` pokes past the left edge or `passband2` past
  the right edge — that window needs a dedicated edge filter. This replaces the cryptic `lo > hi`
  error the scan would otherwise raise.
"""
function validate_passbands_within_spectrum(filter::DoubleDeltaFilter, ε::Float64)
    filter.passband1.left - ε < -1.0 &&
        error("DoubleDeltaFilter: passband1 reaches the left spectrum edge (left=$(filter.passband1.left)); use a dedicated edge filter.")
    filter.passband2.right + ε > 1.0 &&
        error("DoubleDeltaFilter: passband2 reaches the right spectrum edge (right=$(filter.passband2.right)); use a dedicated edge filter.")
    return true
end

"""
    validate_passbands(P, filter, ε, leak_rtol, npts) -> Bool

Check that `P` stays amplified across BOTH passband interiors: above `cutoff_upper` on `passband1`
(the `+1` peak) and below `cutoff_lower` on `passband2` (the `−1` peak). Each interior is shrunk by
`ε` on both sides so the on-cutoff passband edges are not re-flagged.

# Arguments
- `P`: the combined filter polynomial.
- `filter::DoubleDeltaFilter`: supplies the passbands and cutoffs.
- `ε`: boundary margin (passband edges sit exactly on the cutoffs).
- `leak_rtol`: relative slack passed to the one-sided leak test.
- `npts`: samples per passband interior.

# Returns
- `true` if both interiors stay past their cutoff.

# Notes
The boundary margin is adaptive, `min(ε, width/10)` per passband, so the scanned interior is never
empty — a very narrow passband collapses toward its center rather than being skipped.
"""
function validate_passbands(P, filter::DoubleDeltaFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    ε1 = min(ε, (filter.passband1.right - filter.passband1.left) / 10)   # adaptive margin, never empties
    ε2 = min(ε, (filter.passband2.right - filter.passband2.left) / 10)
    pb1 = (filter.passband1.left + ε1, filter.passband1.right - ε1)
    pb2 = (filter.passband2.left + ε2, filter.passband2.right - ε2)

    pb1_ok = validate_no_leakage_in_interval_numerically(P, pb1[1], pb1[2], filter.cutoff_upper, -1; npts = npts, rtol = leak_rtol)
    pb2_ok = validate_no_leakage_in_interval_numerically(P, pb2[1], pb2[2], filter.cutoff_lower, +1; npts = npts, rtol = leak_rtol)

    return pb1_ok && pb2_ok
end

"""
    validate_outer_tails(P, filter, ε, leak_rtol, npts) -> Bool

Check that both outer tails — left of `passband1` and right of `passband2` — keep `P` confined to
the band `[cutoff_lower, cutoff_upper]` (no side lobe crossing either cutoff). Each tail is scanned
twice: coarsely across its whole span, and again over a `2·(passband width)` window right next to
the passband, where side lobes are strongest.

# Arguments
- `P`: the combined filter polynomial.
- `filter::DoubleDeltaFilter`: supplies the passbands and cutoffs.
- `ε`: boundary margin from the passband edges.
- `leak_rtol`, `npts`: forwarded to [`_within_band`](@ref).

# Returns
- `true` iff BOTH tails are clean (the two sides are AND-ed, so a leak in either fails).

# Notes
Because the result is a single boolean over both tails, this cannot on its own distinguish a
left-only from a right-only leak.
"""
function validate_outer_tails(P, filter::DoubleDeltaFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    cl, cu = filter.cutoff_lower, filter.cutoff_upper
    l1     = filter.passband1.left  - ε
    r2     = filter.passband2.right + ε
    pbw1   = filter.passband1.right - filter.passband1.left
    pbw2   = filter.passband2.right - filter.passband2.left

    left_ok =
        _within_band(P, -1.0, l1, cl, cu; npts = npts, rtol = leak_rtol) &&
        _within_band(P, max(-1.0, l1 - 2pbw1), l1, cl, cu; npts = npts, rtol = leak_rtol)
    right_ok =
        _within_band(P, r2, 1.0, cl, cu; npts = npts, rtol = leak_rtol) &&
        _within_band(P, r2, min(1.0, r2 + 2pbw2), cl, cu; npts = npts, rtol = leak_rtol)

    return left_ok && right_ok
end

"""
    validate_between_passbands(P, filter, ε, leak_rtol, npts) -> Bool

Check that the valley between the two passbands — `[passband1.right+ε, passband2.left-ε]` — keeps `P`
confined to the band `[cutoff_lower, cutoff_upper]`. A real valley leak shows up as an INTERIOR
extremum poking past a cutoff (a ringing lobe between the peaks); a clean valley is a monotone
transition whose extrema sit at the passband edges. `npts` scales with order
([`DoubleDeltaFilterDefaults`](@ref) `npts_between_passbands`) so shallow interior lobes of high-order
filters are resolved.

# Returns
- `true` if the valley stays inside the band, `false` on a leak.

# Notes
The boundary margin is adaptive, `min(ε, valley_width/10)`, so the scanned valley is never emptied
and skipped — even when the passbands nearly touch.
"""
function validate_between_passbands(P, filter::DoubleDeltaFilter, ε::Float64, leak_rtol::Float64, npts::Int)
    εv = min(ε, (filter.passband2.left - filter.passband1.right) / 10)   # adaptive margin, never empties
    lo = filter.passband1.right + εv
    hi = filter.passband2.left  - εv
    return _within_band(P, lo, hi, filter.cutoff_lower, filter.cutoff_upper; npts = npts, rtol = leak_rtol)
end
