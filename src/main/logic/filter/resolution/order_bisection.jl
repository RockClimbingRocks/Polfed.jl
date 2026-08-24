# Shared resolution mechanics for the filters. Two generic primitives live here:
#   • `bisect_order`   — resolve the polynomial ORDER: the DRIVER (bracket by doubling, then bisect)
#     is generic; each filter supplies the `suppressed(order)::Bool` PREDICATE ("has P dropped below
#     cutoff across my window?"), so the primitive never needs to know how many endpoints a filter
#     tests (one for a delta, several for a double/edge filter) or which sign its peak is.
#   • `refine_crossing` — resolve a passband EDGE: bisect `P(cos θ) − cutoff` to a guaranteed value
#     accuracy `|P(x) − cutoff| ≤ atol` (order-independent), shared by every filter's passband scan.

"""
    bisect_order(analytical_K::Real, suppressed) -> Int

Smallest integer order at which the filter's `suppressed(order)::Bool` predicate first holds.
The order is bracketed by doubling up from `2.5·analytical_K` until `suppressed` turns true,
then refined by [`bisection`](@ref) to integer resolution.

`suppressed` is the filter's own test that its polynomial has fallen below `cutoff` everywhere
the window requires (oriented toward the filter's exposed end). Keeping it a predicate is what
makes this reusable across single- and multi-window filters — the driver here is filter-agnostic.

Returns `1` immediately if even order 1 already suppresses (a very wide window). The bracket and
the resolved order are traced at DEEP-debug level (the order IS the per-matvec cost multiplier).
"""
function bisect_order(analytical_K::Real, suppressed)
    suppressed(1) && return 1                                  # order 1 already enough

    Kmax = max(2, ceil(Int, analytical_K * 1.75))
    while !suppressed(Kmax)
        Kmax *= 2
        Kmax > 10^10 && error("bisect_order: could not bracket an order below cutoff.")
    end

    # Integer bisection: return the first suppressing order exactly. A real-valued bisection with
    # `floor`/`ceil` can overshoot by one; for narrow intervals that can cancel the safety factor.
    lo, hi = 1, Kmax
    while hi - lo > 1
        mid = lo + (hi - lo) ÷ 2
        if suppressed(mid)
            hi = mid
        else
            lo = mid
        end
    end
    polfed_log(POLFED_DEEP_DEBUG_LEVEL, "Order bisection resolved.",
               analytical_K = analytical_K, bracket = (1, Kmax), order = hi)
    return hi
end

"""
    refine_crossing(residual, θ_lo, θ_hi; atol=1e-10, postol=1e-14, max_iter=200) -> Float64

Bisect `residual(θ) = P(cos θ) − cutoff` for its root in the bracket `[θ_lo, θ_hi]` — which must
contain a sign change — and return the `θ` of the crossing. The angle `θ = acos(x)` is the natural
variable: the Chebyshev oscillations are uniform in `θ`, so the bracket is well conditioned there.

The stopping rule targets the crossing's **value accuracy** rather than only its position: it
refines until BOTH the bracket half-width is below `postol` AND the residual magnitude at the
returned point is within `atol`, so the crossing satisfies `|P(x) − cutoff| ≤ atol` for **any**
polynomial order. Steep, high-order filters (where a tight position tolerance alone would leave the
value error at `≈ order · postol`) simply take a few extra bisection steps; low-order filters still
reach the much smaller value error they already achieve at `postol`, so precision is never lost.

# Arguments
- `residual`: the θ-space residual `θ -> P(cos θ) − cutoff` (opposite signs at the bracket ends).
- `θ_lo`, `θ_hi`: bracketing angles.

# Keyword Arguments
- `atol::Real=1e-10`: value-accuracy ceiling on `|P(x) − cutoff|` at the returned crossing.
- `postol::Real=1e-14`: position floor on the θ half-width (keeps low-order crossings tight).
- `max_iter::Int=200`: safety cap (only approached at astronomically high orders where floating
  point cannot resolve `atol`); returns the best midpoint so far.

# Returns
- The crossing angle `θ` (take `cos(θ)` for the crossing point `x`).
"""
function refine_crossing(residual, θ_lo::Real, θ_hi::Real;
                         atol::Real = 1e-10, postol::Real = 1e-14, max_iter::Int = 200)
    r_lo = residual(θ_lo)
    r_lo == 0 && return float(θ_lo)

    θ_mid = (θ_lo + θ_hi) / 2
    for _ in 1:max_iter
        θ_mid = (θ_lo + θ_hi) / 2
        r_mid = residual(θ_mid)

        # Stop only once the crossing is both located (tight bracket) AND on the cutoff (value ≤ atol).
        ((θ_hi - θ_lo) / 2 < postol && abs(r_mid) <= atol) && return θ_mid
        r_mid == 0 && return θ_mid

        if r_lo * r_mid < 0
            θ_hi = θ_mid
        else
            θ_lo, r_lo = θ_mid, r_mid
        end
    end
    return θ_mid
end
