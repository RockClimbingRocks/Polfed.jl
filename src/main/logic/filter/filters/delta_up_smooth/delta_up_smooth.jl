# DeltaUpFilter — the default POLFED filter (reference implementation of the per-filter layout).
# A delta-peaked Chebyshev polynomial that amplifies a single target energy λ to the HIGHEST end
# of the transformed spectrum (P(λ) = +1) and suppresses everything else below `cutoff`; Lanczos
# then targets the highest eigenvalues. Its lowest-end twin is DeltaDownFilter.
#
# This file: the filter's own defaults, its (mutable) struct + constructor, and the trivial
# type-tag methods. The behavior is split into sibling files, included at the bottom:
#   coefficients.jl — coefficient function (normalized to peak +1)
#   build.jl        — analytical_order, resolve_order!, resolve_bounds_from_order, build_filter
#   validate.jl     — leakage check toward the exposed (highest) end

"""Per-filter defaults, namespaced so `cutoff`/`order`/`order_safety_factor` don't collide
across filters (each filter owns a `<Name>Defaults` module)."""
module DeltaUpSmoothFilterDefaults
    const order               = -1     # sentinel: -1 ⇒ resolve the order from `cutoff`
    const order_safety_factor = 1.0   # < 1 shrinks K ⇒ widens the passband (safety margin)

    "Resolution ladder: cutoff candidates `0.13, 0.14, …, 0.31` (19 rungs, 0.01 step) tried in
    order (first valid wins — sharpest usable filter first; higher rungs rescue windows where a
    sharp peak's side lobe leaks, e.g. near a spectrum edge)."
    const cutoff_ladder = ntuple(i -> round(0.01 + 0.01 * i; digits = 2), 23)

    # validate_filter defaults (mirrors DoubleDeltaFilterDefaults)
    const leak_rtol = 0.0  # relative slack on `cutoff` before a lobe counts as a leak
    const Δε        = 1e-8 # margin off the on-cutoff passband edges
    npts_within_passband(order::Int)        = 1_000 # samples across the passband interior
    npts_on_sides_of_the_passband(order::Int) = 1_000 # samples per tail (coarse and fine windows)
end

"""
    DeltaUpFilter(interval; cutoff, order, order_safety_factor)

Delta-peaked Chebyshev filter mapping the target `λ` to the **highest** end of the transformed
spectrum (`P(λ) = +1`), with the rest suppressed below `cutoff`. The default POLFED filter;
Lanczos then requests the highest eigenpairs. Lowest-end counterpart: [`DeltaDownFilter`](@ref).

Mutable and filled in two phases by [`build_filter`](@ref): constructed from the DoS `interval`
(target = its midpoint, order unresolved), then `order` and `passband` are resolved.

# Fields
- `interval::Interval`: the **requested** window (from the DoS) — sets `target`, drives order
  resolution, and is the window results are filtered to.
- `passband::Interval`: the **effective** exposed window from [`resolve_bounds_from_order`](@ref)
  — the window `[left, right]` where the polynomial sits at/above `cutoff`, resolved
  **asymmetrically** per side (`P` is not symmetric about a non-central target). It tracks the
  requested `interval` closely; used by [`validate_filter`](@ref).
- `target::Float64`: rescaled center `λ ∈ [-1, 1]` (the `interval` midpoint).
- `order::Int`: polynomial order `K`; `-1` until resolved (or a user-fixed override).
- `cutoff::Float64`: suppression threshold outside the window — sets the filter width.
- `order_safety_factor::Float64`: multiplies the bisected order (`< 1` widens the passband so all
  requested eigenpairs stay inside).

Direction is fixed by the type ([`exposed_end`](@ref) is `:highest`); the `+1` peak is applied
directly in [`coefficients`](@ref) — nothing stored to keep in sync.
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
                       cutoff::Real              = DeltaUpSmoothFilterDefaults.cutoff,
                       order::Integer            = DeltaUpSmoothFilterDefaults.order,
                       order_safety_factor::Real = DeltaUpSmoothFilterDefaults.order_safety_factor)
    target = (interval.left + interval.right) / 2
    # passband starts as `interval` (placeholder); build_filter resolves it once the order is known.
    return DeltaUpSmoothFilter(
      interval, 
      interval, 
      target, 
      Int(order),
      Float64(cutoff), 
      Float64(order_safety_factor)
    )
end

"""    expected_krylov_dim(::DeltaUpSmoothFilter, howmany, blocksize, η) -> Int
Predicted Krylov dimension from the empirical delta-filter fit `ceil((20.427·blocksize +
1.696·howmany)·η)`; `η` is the overestimation safety factor."""
expected_krylov_dim(::DeltaUpSmoothFilter, howmany::Int, blocksize::Int, η::Real) =
    ceil(Int, (20.427 * blocksize + 1.696 * howmany) * η)

"""    exposed_end(::DeltaUpSmoothFilter) -> :highest — target maps to the top of the transformed spectrum."""
exposed_end(::DeltaUpSmoothFilter) = :highest

"""    passband(f::DeltaUpSmoothFilter) -> Interval — the resolved effective window `[left, right]`."""
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
