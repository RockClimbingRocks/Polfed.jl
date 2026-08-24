# DeltaDownFilter — the lowest-end twin of DeltaUpFilter. Same delta-peaked Chebyshev shape,
# negated: it maps the target to the LOWEST end of the transformed spectrum (P(λ) = -1) and keeps
# everything else above cutoff; Lanczos then targets the lowest eigenpairs. The ONLY differences
# from delta_up are the peak sign (-1) and `exposed_end` — a clean illustration of how a second
# filter slots into the per-filter layout (copy the folder, flip the sign).

# Defaults for this filter live in the central registry: `Defaults.filters.delta_down`
# (src/main/defaults/filters/delta_down.jl).

"""
    DeltaDownFilter(interval; cutoff, order, order_safety_factor)

Delta-peaked Chebyshev filter mapping the target `λ` to the **lowest** end of the transformed
spectrum (`P(λ) = -1`), with the rest kept above `cutoff`. Highest-end counterpart:
[`DeltaUpFilter`](@ref). Fields and two-phase build are identical to [`DeltaUpFilter`](@ref); only
the peak sign (`-1`, applied in [`coefficients`](@ref)) and [`exposed_end`](@ref) (`:lowest`) differ.
"""
mutable struct DeltaDownFilter <: AbstractFilter
    interval::Interval
    passband::Interval
    target::Float64
    order::Int
    cutoff::Float64
    order_safety_factor::Float64
end

function DeltaDownFilter(interval::Interval;
                         cutoff::Real              = Defaults.filters.delta_down.cutoff,
                         order::Integer            = Defaults.filters.delta_down.order,
                         order_safety_factor::Real = Defaults.filters.delta_down.order_safety_factor)
    target = (interval.left + interval.right) / 2

    return DeltaDownFilter(
        interval, 
        interval, 
        target, 
        Int(order),
        Float64(cutoff), 
        Float64(order_safety_factor)
    )
end

expected_krylov_dim(::DeltaDownFilter, howmany::Int, blocksize::Int, η::Real) =
    ceil(Int, (Defaults.filters.delta_down.krylov_iters_per_block * blocksize +
               Defaults.filters.delta_down.krylov_iters_per_eig   * howmany) * η)

"""    exposed_end(::DeltaDownFilter) -> :lowest — target maps to the bottom of the transformed spectrum."""
exposed_end(::DeltaDownFilter) = :lowest

"""    passband(f::DeltaDownFilter) -> Interval — the resolved effective window `[left, right]`."""
passband(f::DeltaDownFilter) = f.passband

"""    cutoff(f::DeltaDownFilter) -> Float64 — the resolved (negative) suppression cutoff."""
cutoff(f::DeltaDownFilter) = f.cutoff

polynomial_type(::DeltaDownFilter) = :Chebyshev

"""    polynomial_order(filter::DeltaDownFilter) -> Int — highest degree of the filter polynomial."""
polynomial_order(filter::DeltaDownFilter) = filter.order

"""    order_safety_factor(filter::DeltaDownFilter) -> Float64 — the resolved order's safety factor."""
order_safety_factor(filter::DeltaDownFilter) = filter.order_safety_factor

include("coefficients.jl")
include("build.jl")
include("validate.jl")
