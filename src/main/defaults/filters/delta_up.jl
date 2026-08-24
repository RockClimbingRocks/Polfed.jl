"""
    UniPolfed.Defaults.filters.delta_up

Defaults for [`DeltaUpFilter`](@ref) — the reference delta filter. Referenced qualified
(`Defaults.filters.delta_up.cutoff`) so `cutoff`/`order`/… never collide across filters.
"""
module delta_up
    const cutoff              = 0.17
    const order               = -1     # sentinel: -1 ⇒ resolve the order from `cutoff`
    const order_safety_factor = 0.97   # < 1 shrinks K ⇒ widens the passband (safety margin)

    "Resolution ladder: cutoff candidates `0.13, 0.14, …, 0.31` (19 rungs, 0.01 step) tried in
    order (first valid wins — sharpest usable filter first; higher rungs rescue windows where a
    sharp peak's side lobe leaks, e.g. near a spectrum edge)."
    const cutoff_ladder = ntuple(i -> round(0.13 + 0.01 * i; digits = 2), 23)

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

    # Krylov-dimension estimate `ceil((iters_per_block·bs + iters_per_eig·howmany)·η)`. An
    # EMPIRICAL FIT from real runs (contrast `Defaults.filters.edge`, whose 25.0/4.0 are untuned
    # initial guesses). The Krylov dimension is driven by `howmany` and the block size, not by the
    # filter's shape, so all four delta-family filters carry the same pair.
    const krylov_iters_per_block = 20.427
    const krylov_iters_per_eig   = 1.696
end
