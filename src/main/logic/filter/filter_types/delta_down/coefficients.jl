# DeltaDownFilter coefficients: identical delta-peaked Chebyshev expansion as delta_up, normalized
# so the polynomial peaks at exactly -1 at the target (the lowest-end peak).

"""    _coefficients_nonnormalized(filter::DeltaDownFilter, n) -> Float64
Raw Chebyshev coefficient `cₙ = (2 - [n==0])·cos(n·acos λ)` (same as the up filter; the sign
enters only in [`coefficients`](@ref))."""
_coefficients_nonnormalized(filter::DeltaDownFilter, n::Int) =
    (2 - (n == 0)) * cos(n * acos(filter.target))

"""
    coefficients(filter::DeltaDownFilter) -> Function

Normalized coefficients `n -> cₙ` rescaled so the polynomial peaks at exactly `-1` at
`filter.target` — the same construction as [`DeltaUpFilter`](@ref)'s, negated. Requires a resolved
`filter.order`.
"""
function coefficients(filter::DeltaDownFilter)
    order = filter.order::Int
    raw = ClenshawExpansion(polynomial_type(filter), n -> _coefficients_nonnormalized(filter, n), order)
    norm_factor = -1.0 / raw(filter.target)                   # -1 peak (lowest end)
    return n -> _coefficients_nonnormalized(filter, n) * norm_factor
end
