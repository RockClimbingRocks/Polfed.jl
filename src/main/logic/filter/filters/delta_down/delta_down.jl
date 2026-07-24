# DeltaDownFilter — the lowest-end twin of DeltaUpFilter. Same delta-peaked Chebyshev shape,
# negated: it maps the target to the LOWEST end of the transformed spectrum (P(λ) = -1) and keeps
# everything else above cutoff; Lanczos then targets the lowest eigenpairs. The ONLY differences
# from delta_up are the peak sign (-1) and `exposed_end` — a clean illustration of how a second
# filter slots into the per-filter layout (copy the folder, flip the sign).

"""Per-filter defaults (own module ⇒ no collision with other filters' `cutoff`/`order`/…)."""
module DeltaDownFilterDefaults
    const cutoff              = -0.17
    const order               = -1
    const order_safety_factor = 0.97

    "Resolution ladder: the NEGATED delta ladder `-0.17, -0.18, …, -0.35` (the `-1` peak keeps
    everything ABOVE a negative cutoff), tried in order — first valid wins."
    const cutoff_ladder = ntuple(i -> -round(0.16 + 0.01 * i; digits = 2), 19)

    # validate_filter defaults (mirrors DoubleDeltaFilterDefaults)
    const leak_rtol = 0.0  # relative slack on `cutoff` before a lobe counts as a leak
    const Δε        = 1e-8 # margin off the on-cutoff passband edges
    npts_within_passband(order::Int)        = 1_000 # samples across the passband interior
    npts_on_sides_of_the_passband(order::Int) = 1_000 # samples per tail (coarse and fine windows)
end

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
                         cutoff::Real              = DeltaDownFilterDefaults.cutoff,
                         order::Integer            = DeltaDownFilterDefaults.order,
                         order_safety_factor::Real = DeltaDownFilterDefaults.order_safety_factor)
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
    ceil(Int, (20.427 * blocksize + 1.696 * howmany) * η)

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
