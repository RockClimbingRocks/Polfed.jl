# DeltaUpSmoothFilter coefficients: the delta-peaked Chebyshev expansion multiplied by a Jackson
# kernel, then normalized so the polynomial peaks at exactly +1 at the target.

"""
    _jackson_kernel(n::Int, order::Int) -> Float64

The Jackson kernel factor `gₙ` for a Chebyshev series truncated at `order` (so `N = order + 1`
terms):

```math
gₙ = \\frac{(N-n+1)\\cos\\frac{πn}{N+1} + \\sin\\frac{πn}{N+1}\\cot\\frac{π}{N+1}}{N+1}
```

`g₀ = 1` and `gₙ` decays to `0` at `n = order`, which is what removes the Gibbs ringing of the
truncated series. This is the same kernel the KPM density-of-states estimate uses — see
[`kpm_density`](@ref) — applied here to a filter expansion instead of a DoS expansion.
"""
function _jackson_kernel(n::Int, order::Int)
    N = order + 1
    q = π / (N + 1)
    return ((N - n + 1) * cos(n * q) + sin(n * q) / tan(q)) / (N + 1)
end

"""
    _coefficients_nonnormalized(filter::DeltaUpSmoothFilter, n::Int) -> Float64

Raw damped Chebyshev coefficient `cₙ = gₙ · (2 - [n==0]) · cos(n · acos λ)`, with `gₙ` the Jackson
factor from [`_jackson_kernel`](@ref). Identical to the sharp filter's coefficient apart from that
factor. Requires a resolved `filter.order` (the kernel depends on the truncation point).
"""
_coefficients_nonnormalized(filter::DeltaUpSmoothFilter, n::Int) =
    _jackson_kernel(n, filter.order) * (2 - (n == 0)) * cos(n * acos(filter.target))

"""
    coefficients(filter::DeltaUpSmoothFilter) -> Function

The filter's **normalized** coefficients `n -> cₙ`, rescaled so the damped polynomial peaks at
exactly `+1` at `filter.target`. Requires a resolved `filter.order`.
"""
function coefficients(filter::DeltaUpSmoothFilter)
    order = filter.order::Int
    raw = ClenshawExpansion(polynomial_type(filter), n -> _coefficients_nonnormalized(filter, n), order)
    norm_factor = 1.0 / raw(filter.target)
    return n -> _coefficients_nonnormalized(filter, n) * norm_factor
end
