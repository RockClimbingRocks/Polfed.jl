"""
    UniPolfed.Defaults.filters.edge

Defaults for [`EdgeFilter`](@ref) — the monotone monomial used for spectrum-edge windows.
"""
module edge
    "Default monomial power `p`. Each Lanczos matvec costs `p` matvecs of `H̃`; higher `p`
    stretches the edge more (fewer iterations) at more matvecs per iteration."
    const order = 3

    "Grid size of the `validate_filter` monotonicity/endpoint sanity scan."
    const npts_validate = 513

    # Krylov-dimension estimate `ceil((iters_per_block·bs + iters_per_eig·howmany)·η)`.
    # INITIAL estimates, not an empirical fit (unlike the delta constants 20.427/1.696): the edge
    # filter's polynomial amplification is weaker than a delta peak, so `iters_per_eig` is set
    # higher. Refine from real runs when data accumulates.
    const krylov_iters_per_block = 25.0
    const krylov_iters_per_eig   = 4.0
    "Extra budget factor when `:both` ends converge in one Krylov space (cf. DoubleDelta's 1.30)."
    const krylov_both_factor     = 1.30
end
