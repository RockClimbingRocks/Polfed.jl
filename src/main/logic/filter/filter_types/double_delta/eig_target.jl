# Lanczos targets for the DoubleDeltaFilter. These live here rather than in filter.jl because the
# generic `resolve_eig_target`/`window_eig_target` cannot express the one thing that matters for a
# double filter: WHICH transformed extreme each window maps to.
#
# The convention is fixed by the filter itself (see `coefficients`): `interval1` — the LOWER-energy
# window — is the `+1` peak, so it maps to the HIGH end of the transformed spectrum, and
# `interval2` — the HIGHER-energy window — is the `−1` peak, mapping to the LOW end. Every target
# built here therefore CROSSES the two arguments: the low `EndSpec` describes window 2, the high
# one describes window 1. Getting this backwards silently returns the wrong window's eigenpairs,
# so it is spelled out once, here, and nowhere else.

"""
    double_window_eig_target(filter::DoubleDeltaFilter, interval1, interval2) -> BothEnds

The Lanczos [`EigTarget`](@ref) for a double-filter **window** run: one [`Window`](@ref) per
requested window, placed at the transformed extreme that window maps to —
`BothEnds(Window(interval2), Window(interval1))`.

Both windows are stated in **rescaled** coordinates (the space `interval1`/`interval2` and the
filter live in), so the caller must pass a rescaled `mapvals` and a rescaled `eigentol`, exactly
as the single-filter window pipeline does; the un-rescaling happens once, at the result boundary.

A bound sitting on a rescaled spectrum edge (`±1`) is encoded as `∓Inf`, which the
[`Window`](@ref) criterion reads as "no bracket exists on that side" — the same encoding
[`window_eig_target`](@ref) uses. In practice a double filter never sees an edge-touching window
(those are routed to an [`EdgeFilter`](@ref) and hence to two separate runs), but the encoding
costs nothing and keeps the two paths consistent.

# Arguments
- `filter::DoubleDeltaFilter`: the resolved double filter (only its convention is used).
- `interval1::Interval`: the LOWER-energy window (the `+1` peak → transformed HIGH end).
- `interval2::Interval`: the HIGHER-energy window (the `−1` peak → transformed LOW end).

# Returns
- [`BothEnds`](@ref) with the low end targeting `interval2` and the high end `interval1`.

# Notes
`return_brackets` is left at its default (`true`): each window is returned bracketed, so the
caller can see the interval is enclosed. The bracket on a window's INNER side comes from the
valley between the two peaks — the pipeline limits that widening so one window can never adopt
the other window's eigenpair as its bracket (see `build_polfed_result`'s `bracket_window`).
"""
double_window_eig_target(::DoubleDeltaFilter, interval1::Interval, interval2::Interval) =
    BothEnds(
        Window(interval2.left  <= -1.0 + 1e-9 ? -Inf : interval2.left,
               interval2.right >=  1.0 - 1e-9 ?  Inf : interval2.right),
        Window(interval1.left  <= -1.0 + 1e-9 ? -Inf : interval1.left,
               interval1.right >=  1.0 - 1e-9 ?  Inf : interval1.right),
    )

"""
    double_count_eig_target(filter::DoubleDeltaFilter, howmany1, howmany2) -> BothEnds

The Lanczos [`EigTarget`](@ref) for a double-filter **count** run: `howmany1` eigenpairs from the
transformed HIGH end (window 1, the `+1` peak) and `howmany2` from the LOW end (window 2, the
`−1` peak) — i.e. `BothEnds(howmany2, howmany1)`, since [`BothEnds`](@ref)`(n_low, n_high)` takes
the low end first.

Contrast [`resolve_eig_target`](@ref)`(:both, howmany)`, which is the symmetric shorthand
(`howmany` from *each* extreme) and cannot express two different counts.
"""
double_count_eig_target(::DoubleDeltaFilter, howmany1::Integer, howmany2::Integer) =
    BothEnds(Int(howmany2), Int(howmany1))
