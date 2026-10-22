"""
    Polfed.Defaults.filters.double_delta

Defaults for [`DoubleDeltaFilter`](@ref) — two peaks (`+1` / `-1`) in one polynomial.
"""
module double_delta
    const cutoff_lower        = -0.25
    const cutoff_upper        = 0.25
    const order               = -1     # sentinel: -1 ⇒ resolve the order from `cutoff`
    const order_safety_factor = 0.97   # < 1 shrinks K ⇒ widens the passband (safety margin)

    "Resolution ladder: symmetric cutoff magnitudes `0.17, 0.18, …, 0.35`, each consumed as
    `cutoff_upper = +c` / `cutoff_lower = -c` — first valid wins."
    const cutoff_ladder = ntuple(i -> round(0.16 + 0.01 * i; digits = 2), 19)

    const leak_rtol = 0.0 # relative tolerance how large can the leakage be for a given cutoff
    const Δε = 1e-8 # small number to guard the passband edges
    npts_within_passband(order1::Int, order2::Int) = 1_000 # number of points to sample within the passband for leakage check
    npts_on_sides_of_the_passband(order1::Int, order2::Int) = 1_000 # number of points to sample on the sides of the passband for leakage check
    npts_between_passbands(order1::Int, order2::Int) = Int(10*max(order1, order2)) # number of points to sample between the passbands for leakage check

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
