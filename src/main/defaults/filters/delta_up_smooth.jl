"""
    Polfed.Defaults.filters.delta_up_smooth

Defaults for [`DeltaUpSmoothFilter`](@ref) — the Jackson-damped fallback, last in the
interior chain.
"""
module delta_up_smooth
    const cutoff              = 0.17
    const order               = -1     # sentinel: -1 ⇒ resolve the order from `cutoff`
    const order_safety_factor = 0.97   # < 1 shrinks K ⇒ widens the passband (safety margin)

    """Bracket constant for [`analytical_order`](@ref). Jackson damping roughly doubles the width
    of the main peak at a given order, so the seed is about twice the delta filter's `2.65499`.
    Only used to *bracket* the exact bisection, so it need not be sharp."""
    const order_bracket_constant = 5.31

    "Resolution ladder: the same cutoff rungs as the sharp delta filters, sharpest first."
    const cutoff_ladder = ntuple(i -> round(0.13 + 0.01 * i; digits = 2), 23)

    # validate_filter defaults (mirrors Defaults.filters.delta_up)
    """Reject a candidate whose passband covers this fraction of `[-1, 1]` or more. The damped
    peak is wide, so at a low order it can sit above `cutoff` across almost the whole spectrum —
    technically "valid" (it covers the request, it does not leak) while suppressing nothing.
    That is the degenerate near-constant filter. A candidate admitting 90% of the spectrum is
    not filtering anything — such a request belongs to `patched_polfed`, `full_polfed`, or plain
    ED, and the chain should exhaust and say so rather than hand back something useless. The
    sharp filters reach the same verdict by a different route: their passband runs into `±1` and
    the strictly-interior check rejects them."""
    const max_passband_fraction = 0.90

    const leak_rtol = 0.0  # relative slack on `cutoff` before a lobe counts as a leak
    const Δε        = 1e-8 # margin off the on-cutoff passband edges
    npts_within_passband(order::Int)          = 1_000
    npts_on_sides_of_the_passband(order::Int) = 1_000

    # Krylov-ITERATION fit `iters(x) = A·x^α·(1 + x/xc)^(1-α)` at `x = howmany/blocksize`; the
    # Krylov DIMENSION is `iters(x)·blocksize·η` (see `_fitted_krylov_dim`). An EMPIRICAL FIT from
    # real runs (contrast `Defaults.filters.edge`, whose 25.0/4.0 are untuned initial guesses).
    # One crossover, not a line: sublinear (`≈ A·x^α`) below `xc ≈ 50` eigenpairs per block,
    # straightening to `≈ 1.63` dimensions per eigenpair above it. The Krylov dimension is driven
    # by `howmany` and the block size, not by the filter's shape, so all four delta-family filters
    # carry the same triple.
    const krylov_fit_A     = 16.725632
    const krylov_fit_alpha = 0.40598213
    const krylov_fit_xc    = 50.265139
end
