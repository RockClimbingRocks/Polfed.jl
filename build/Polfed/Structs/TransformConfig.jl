"""
    TransformConfig(; kwargs...)

Configuration for polynomial transform settings (coefficients, order, interval).
"""
mutable struct TransformConfig
    coefficients::Function
    normalization::Real
    cutoff::Real
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    order::Union{Int, Nothing}
    order_safety_factor::Real
    polynomialtype::Symbol

    """Build a `TransformConfig` from keyword arguments."""
    function TransformConfig(;
        coefficients::Function        = PolfedDefaults.coefficients,
        normalization::Real           = PolfedDefaults.normalization,
        cutoff::Real                  = PolfedDefaults.cutoff,
        left::Union{Real, Nothing}    = nothing,
        right::Union{Real, Nothing}   = nothing,
        order::Union{Int, Nothing}    = nothing,
        order_safety_factor::Real     = PolfedDefaults.order_safety_factor,
        polynomialtype::Symbol        = PolfedDefaults.polynomialtype,
    )
        new(coefficients, normalization, cutoff, left, right, order, order_safety_factor, polynomialtype)
    end
end
