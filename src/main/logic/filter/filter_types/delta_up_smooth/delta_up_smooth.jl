# DeltaUpSmoothFilter — a DeltaUpFilter whose Chebyshev coefficients are damped by a Jackson
# kernel before normalization. Same geometry (single `+1` peak at the target, exposed end
# `:highest`), same resolution machinery — only the coefficient function differs.
#
# Why it exists: a raw delta expansion is a truncated series, so it rings. Those Gibbs side lobes
# are what `validate_filter` rejects when a sharp candidate leaks, and the delta chain's answer is
# to walk to a laxer cutoff. Damping removes the ringing at the source instead. The price is a
# WIDER main peak at the same order, so the same window costs a higher `K` — which is why this
# filter sits LAST in the interior chain: it is the fallback for windows the sharp filters cannot
# serve cleanly, not the default.
#
# This file: defaults, the (mutable) struct + constructor, and the type-tag methods. Behavior is
# split into sibling files, included at the bottom:
#   coefficients.jl — Jackson-damped coefficient function (normalized to peak +1)
#   build.jl        — analytical_order, resolve_order!, resolve_bounds_from_order!, build_filter
#   validate.jl     — leakage check toward the exposed (highest) end

# Defaults for this filter live in the central registry: `Defaults.filters.delta_up_smooth`
# (src/main/defaults/filters/delta_up_smooth.jl).

"""
    DeltaUpSmoothFilter(interval; cutoff, order, order_safety_factor)

Jackson-damped delta filter mapping the target `λ` to the **highest** end of the transformed
spectrum (`P(λ) = +1`). Geometrically identical to [`DeltaUpFilter`](@ref) — one peak, exposed
end `:highest` — but its Chebyshev coefficients are multiplied by a Jackson kernel before
normalization, which suppresses the Gibbs side lobes of the truncated series.

Trade-off: the damped main peak is roughly **twice as wide** at a given order, so isolating the
same window needs roughly **twice the order** — and order is what every matvec pays for. It is
therefore placed last in [`Defaults.filters.INTERIOR_FILTER_CHAIN`](@ref): tried only when the sharp delta filters
cannot produce a leak-free candidate for the requested window.

Mutable and filled in two phases by [`build_filter`](@ref), exactly like the sharp filters.

# Fields
- `interval::Interval`: the **requested** window (from the DoS).
- `passband::Interval`: the **effective** exposed window, resolved asymmetrically per side.
- `target::Float64`: rescaled center `λ ∈ [-1, 1]` (the `interval` midpoint).
- `order::Int`: polynomial order `K`; `-1` until resolved.
- `cutoff::Float64`: suppression threshold outside the window.
- `order_safety_factor::Float64`: multiplies the bisected order (`< 1` widens the passband).

See also [`DeltaUpFilter`](@ref), [`DeltaDownFilter`](@ref).
"""
mutable struct DeltaUpSmoothFilter <: AbstractFilter
    interval::Interval
    passband::Interval
    target::Float64
    order::Int
    cutoff::Float64
    order_safety_factor::Float64
end

function DeltaUpSmoothFilter(interval::Interval;
                             cutoff::Real              = Defaults.filters.delta_up_smooth.cutoff,
                             order::Integer            = Defaults.filters.delta_up_smooth.order,
                             order_safety_factor::Real = Defaults.filters.delta_up_smooth.order_safety_factor)
    target = (interval.left + interval.right) / 2
    return DeltaUpSmoothFilter(
        interval,
        interval,
        target,
        Int(order),
        Float64(cutoff),
        Float64(order_safety_factor),
    )
end

"""    expected_krylov_dim(::DeltaUpSmoothFilter, howmany, blocksize, η) -> Int
Same empirical fit as the sharp delta filters — the Krylov dimension is driven by `howmany` and
the block size, not by the filter's shape."""
expected_krylov_dim(::DeltaUpSmoothFilter, howmany::Int, blocksize::Int, η::Real) =
    ceil(Int, (Defaults.filters.delta_up_smooth.krylov_iters_per_block * blocksize +
               Defaults.filters.delta_up_smooth.krylov_iters_per_eig   * howmany) * η)

"""    exposed_end(::DeltaUpSmoothFilter) -> :highest — target maps to the top of the transformed spectrum."""
exposed_end(::DeltaUpSmoothFilter) = :highest

"""    passband(f::DeltaUpSmoothFilter) -> Interval — the resolved effective window."""
passband(f::DeltaUpSmoothFilter) = f.passband

"""    cutoff(f::DeltaUpSmoothFilter) -> Float64 — the resolved suppression cutoff."""
cutoff(f::DeltaUpSmoothFilter) = f.cutoff

"""    polynomial_type(::DeltaUpSmoothFilter) -> :Chebyshev — the filter's polynomial family."""
polynomial_type(::DeltaUpSmoothFilter) = :Chebyshev

"""    polynomial_order(filter::DeltaUpSmoothFilter) -> Int — highest degree of the filter polynomial."""
polynomial_order(filter::DeltaUpSmoothFilter) = filter.order

"""    order_safety_factor(filter::DeltaUpSmoothFilter) -> Float64 — the resolved order's safety factor."""
order_safety_factor(filter::DeltaUpSmoothFilter) = filter.order_safety_factor

include("coefficients.jl")
include("build.jl")
include("validate.jl")
