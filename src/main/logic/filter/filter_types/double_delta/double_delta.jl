



mutable struct DoubleDeltaFilter <: AbstractFilter
    interval1::Interval
    passband1::Interval
    target1::Float64
    order1::Int
    
    interval2::Interval
    passband2::Interval
    target2::Float64
    order2::Int

    cutoff_lower::Float64
    cutoff_upper::Float64
    order_safety_factor::Float64
end


# Defaults for this filter live in the central registry: `Defaults.filters.double_delta`
# (src/main/defaults/filters/double_delta.jl).




function DoubleDeltaFilter(interval1::Interval, interval2::Interval;
                           cutoff_lower::Real           = Defaults.filters.double_delta.cutoff_lower,
                           cutoff_upper::Real           = Defaults.filters.double_delta.cutoff_upper,
                           order1::Int              = Defaults.filters.double_delta.order,
                           order2::Int              = Defaults.filters.double_delta.order,
                           order_safety_factor::Real    = Defaults.filters.double_delta.order_safety_factor)

    interval1.left < interval1.right ||
        throw(ArgumentError("DoubleDeltaFilter: interval1 must be non-empty (left < right); got [$(interval1.left), $(interval1.right)]."))
    interval2.left < interval2.right ||
        throw(ArgumentError("DoubleDeltaFilter: interval2 must be non-empty (left < right); got [$(interval2.left), $(interval2.right)]."))
    interval1.right < interval2.left ||
        throw(ArgumentError("DoubleDeltaFilter: interval1 must lie strictly left of interval2; got [$(interval1.left), $(interval1.right)] and [$(interval2.left), $(interval2.right)]."))

    target1 = (interval1.left + interval1.right) / 2
    target2 = (interval2.left + interval2.right) / 2

    return DoubleDeltaFilter(
        interval1, 
        interval1, 
        target1, 
        order1,
        interval2, 
        interval2, 
        target2, 
        order2,
        Float64(cutoff_lower), 
        Float64(cutoff_upper), 
        Float64(order_safety_factor),
    )
end

"""
    expected_krylov_dim(::DoubleDeltaFilter, howmany, blocksize, η) -> Int

Predicted Krylov dimension for the JOINT run: the empirical delta-filter fit
`A·x^α·(1 + x/xc)^(1-α)` iterations at `x = howmany/blocksize` ([`_fitted_krylov_dim`](@ref),
constants in [`Defaults.filters.double_delta`](@ref)), inflated by
[`Defaults.double_polfed.krylov_overhead`](@ref) (`1.4`) and the caller's overestimation factor
`η`.

**`howmany` is per END, not the total.** One `BothEnds` Krylov space converges both transformed
extremes in the same factorization, so the budget scales with `max(howmany₁, howmany₂)` — the
harder of the two windows — not with their sum; the overhead constant is what pays for serving
the second end. Passing the sum here would over-budget the run and make the `:auto` strategy
decision inconsistent with [`Defaults.double_polfed.double_ratio`](@ref), which is derived from
the same constant.
"""
expected_krylov_dim(::DoubleDeltaFilter, howmany::Int, blocksize::Int, η::Real) =
    _fitted_krylov_dim(howmany, blocksize,
                       Defaults.double_polfed.krylov_overhead * η,
                       Defaults.filters.double_delta.krylov_fit_A,
                       Defaults.filters.double_delta.krylov_fit_alpha,
                       Defaults.filters.double_delta.krylov_fit_xc)

"""    cutoff(filter::DoubleDeltaFilter) -> Float64 — the peak cutoff magnitude (the ladder rung `c`
that set `cutoff_upper = +c` / `cutoff_lower = −c`). Read by the report."""
cutoff(filter::DoubleDeltaFilter) = filter.cutoff_upper

"""
    filter_target(filter::DoubleDeltaFilter, interval::Interval) -> Float64

Which of the two peaks (`target1` → `+1`, `target2` → `−1`) serves `interval`: the nearer one to
the interval's center. A double filter has no single `.target` field, so the report asks through
this accessor — one `SpectralTransformReport` is built per window, each naming its own peak.
"""
filter_target(filter::DoubleDeltaFilter, interval::Interval) =
    let center = (interval.left + interval.right) / 2
        abs(filter.target1 - center) <= abs(filter.target2 - center) ? filter.target1 :
                                                                       filter.target2
    end

"""    exposed_end(::DoubleDeltaFilter) -> :highest — target maps to the top of the transformed spectrum."""
exposed_end(::DoubleDeltaFilter) = :both

"""    polynomial_type(::DeltaUpFilter) -> :Chebyshev — the filter's polynomial family."""
polynomial_type(::DoubleDeltaFilter) = :Chebyshev

"""
    polynomial_order(filter::DoubleDeltaFilter) -> Int

Highest degree of the combined double-delta polynomial. Each lobe has its own
order, and the combined coefficient sequence spans the larger one.
"""
polynomial_order(filter::DoubleDeltaFilter) = max(filter.order1, filter.order2)

"""    order_safety_factor(filter::DoubleDeltaFilter) -> Float64 — the resolved orders' safety factor."""
order_safety_factor(filter::DoubleDeltaFilter) = filter.order_safety_factor

include("coefficients.jl")
include("build.jl")
include("validate.jl")
include("eig_target.jl")   # the :both Lanczos targets (window + count), incl. the ±1/window flip
