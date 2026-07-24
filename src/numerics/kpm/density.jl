
function kpm_coeffs(moments::AbstractVector{<:Real}, kernel::Symbol)
    N = length(moments)

    g = get_kernel(kernel)
    coeffs = [g(k, N) * moments[k + 1] * (k == 0 ? 1 : 2) for k in 0:N-1]
    return coeffs
end


"""
    kpm_density(moments; kernel=:Jackson) -> Function

Reconstruct the rescaled spectral density from KPM Chebyshev moments.

The returned `rho(x)` is valid on the open interval `x in (-1, 1)`.
"""
function kpm_density(moments::AbstractVector{<:Real}; kernel = :Jackson)
    N = length(moments)
    N >= 1 || throw(ArgumentError("kpm_density requires at least one moment."))

    coeffs = kpm_coeffs(moments, kernel)
    expansion = ClenshawExpansion(:Chebyshev, coeffs, N - 1)

    return x -> expansion(x) / (pi * sqrt(1 - x^2))
end
