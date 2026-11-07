"""
    Polfed.Defaults.filters.delta_down

Defaults for [`DeltaDownFilter`](@ref) — the lowest-end twin of `delta_up`, sign flipped.
"""
module delta_down
    const cutoff              = -0.17
    const order               = -1
    const order_safety_factor = 0.97

    "Resolution ladder: the NEGATED delta ladder `-0.17, -0.18, …, -0.35` (the `-1` peak keeps
    everything ABOVE a negative cutoff), tried in order — first valid wins."
    const cutoff_ladder = ntuple(i -> -round(0.16 + 0.01 * i; digits = 2), 19)

    # validate_filter defaults (mirrors Defaults.filters.double_delta)
    const leak_rtol = 0.0  # relative slack on `cutoff` before a lobe counts as a leak
    const Δε        = 1e-8 # margin off the on-cutoff passband edges
    npts_within_passband(order::Int)        = 1_000 # samples across the passband interior
    npts_on_sides_of_the_passband(order::Int) = 1_000 # samples per tail (coarse and fine windows)

    """Bracket constant `c` in `K ≈ c·√(1-λ²)/δ`, the heuristic seed for [`analytical_order`](@ref).
    Delta-filter-specific and empirical. Used only to *bracket* the exact [`bisect_order`](@ref),
    which doubles outward until the cutoff is cleared — an imperfect seed costs a couple of extra
    probes, nothing more. `delta_up_smooth`'s is about twice this: Jackson damping widens the main
    peak at a given order."""
    const order_bracket_constant = 2.65499

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
