# DeltaUpFilter coefficients: the delta-peaked Chebyshev expansion, normalized so the polynomial
# peaks at exactly +1 at the target (the highest-end peak).

"""
    _coefficients_nonnormalized(filter::DeltaUpSmoothFilter, n::Int) -> Float64

Raw (unnormalized) Chebyshev coefficient `cₙ` of the delta-peaked filter centered at
`λ = filter.target`: `cₙ = (2 - [n==0]) · cos(n · acos λ)`, so `Σₙ cₙ Tₙ` peaks at `λ`. Internal
building block — only [`coefficients`](@ref) calls it, to rescale the peak to `+1`.
"""
_coefficients_nonnormalized(filter::DeltaUpSmoothFilter, n::Int) =
    (2 - (n == 0)) * cos(n * acos(filter.target))


    
function _damping_coeffs(order, n)
    θ = pi / (order + 1)
    cotθ = 1.0 / tan(θ)

    return ((order - n + 1) * cos(n * θ) + sin(n * θ) * cotθ) / (order + 1)
end




"""
    coefficients(filter::DeltaUpSmoothFilter) -> Function

The filter's **normalized** coefficients `n -> cₙ`, rescaled so the polynomial peaks at exactly
`+1` at `filter.target`: the raw expansion (to `filter.order`) evaluated at the target is its peak
height, and dividing by it sends the peak to `+1`. This is the coefficient function the
transform/resolution machinery consumes. Requires a resolved `filter.order`.
"""
function coefficients(filter::DeltaUpSmoothFilter)
    order = filter.order::Int
    raw = ClenshawExpansion(polynomial_type(filter), n -> _coefficients_nonnormalized(filter, n) * _damping_coeffs(order, n), order)
    norm_factor = 1.0 / raw(filter.target)                    # +1 peak (highest end)
    return n -> _coefficients_nonnormalized(filter, n) * _damping_coeffs(norm_factor, n) * norm_factor
end