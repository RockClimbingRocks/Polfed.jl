



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


module DoubleDeltaFilterDefaults
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

end




function DoubleDeltaFilter(interval1::Interval, interval2::Interval;
                           cutoff_lower::Real           = DoubleDeltaFilterDefaults.cutoff_lower,
                           cutoff_upper::Real           = DoubleDeltaFilterDefaults.cutoff_upper,
                           order1::Int              = DoubleDeltaFilterDefaults.order,
                           order2::Int              = DoubleDeltaFilterDefaults.order,
                           order_safety_factor::Real    = DoubleDeltaFilterDefaults.order_safety_factor)

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

"""    expected_krylov_dim(::DoubleDeltaFilter, howmany, blocksize, η) -> Int
Predicted Krylov dimension from the empirical delta-filter fit `ceil((20.427·blocksize +
1.696·howmany)·η)`; `η` is the overestimation safety factor."""
expected_krylov_dim(::DoubleDeltaFilter, howmany::Int, blocksize::Int, η::Real) =
    ceil(Int, (20.427 * blocksize + 1.696 * howmany) * 1.30 * η)

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
