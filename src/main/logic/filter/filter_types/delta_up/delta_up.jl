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

# Defaults for this filter live in the central registry: `Defaults.filters.delta_up`
# (src/main/defaults/filters/delta_up.jl).

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
mutable struct DeltaUpFilter <: AbstractFilter
    interval::Interval
    passband::Interval
    target::Float64
    order::Int
    cutoff::Float64
    order_safety_factor::Float64
end

function DeltaUpFilter(interval::Interval;
                       cutoff::Real              = Defaults.filters.delta_up.cutoff,
                       order::Integer            = Defaults.filters.delta_up.order,
                       order_safety_factor::Real = Defaults.filters.delta_up.order_safety_factor)
    target = (interval.left + interval.right) / 2
    # passband starts as `interval` (placeholder); build_filter resolves it once the order is known.
    return DeltaUpFilter(
      interval, 
      interval, 
      target, 
      Int(order),
      Float64(cutoff), 
      Float64(order_safety_factor)
    )
end

"""    expected_krylov_dim(::DeltaUpFilter, howmany, blocksize, η) -> Int
Predicted Krylov dimension from the empirical delta-filter fit: `A·x^α·(1 + x/xc)^(1-α)`
iterations at `x = howmany/blocksize`, times `blocksize` and `η` (the overestimation safety
factor). See [`_fitted_krylov_dim`](@ref); constants in [`Defaults.filters.delta_up`](@ref)."""
expected_krylov_dim(::DeltaUpFilter, howmany::Int, blocksize::Int, η::Real) =
    _fitted_krylov_dim(howmany, blocksize, η,
                       Defaults.filters.delta_up.krylov_fit_A,
                       Defaults.filters.delta_up.krylov_fit_alpha,
                       Defaults.filters.delta_up.krylov_fit_xc)

"""    exposed_end(::DeltaUpFilter) -> :highest — target maps to the top of the transformed spectrum."""
exposed_end(::DeltaUpFilter) = :highest

"""    passband(f::DeltaUpFilter) -> Interval — the resolved effective window `[left, right]`."""
passband(f::DeltaUpFilter) = f.passband

"""    cutoff(f::DeltaUpFilter) -> Float64 — the resolved suppression cutoff."""
cutoff(f::DeltaUpFilter) = f.cutoff

"""    polynomial_type(::DeltaUpFilter) -> :Chebyshev — the filter's polynomial family."""
polynomial_type(::DeltaUpFilter) = :Chebyshev

"""    polynomial_order(filter::DeltaUpFilter) -> Int — highest degree of the filter polynomial."""
polynomial_order(filter::DeltaUpFilter) = filter.order

"""    order_safety_factor(filter::DeltaUpFilter) -> Float64 — the resolved order's safety factor."""
order_safety_factor(filter::DeltaUpFilter) = filter.order_safety_factor

include("coefficients.jl")
include("build.jl")
include("validate.jl")
