"""
    integrate_chebyshev_weighted(coefficients, a, b) -> Real

Exact definite integral of a Chebyshev series against the Chebyshev weight,

    ∫ₐᵇ [ Σₖ cₖ Tₖ(x) ] / √(1 - x²) dx ,

where `coefficients[k+1] = cₖ` for `k = 0 … length(coefficients)-1`.

# Why the weighted integral (and not `∫ Tₖ dx`)

The KPM spectral density carries exactly this weight,

    ρ(x) = [ Σₖ cₖ Tₖ(x) ] / (π √(1 - x²)) ,

so the number of states in a rescaled window `[a, b]` is `∫ₐᵇ ρ = (1/π)·` this
integral. Integrating *against the weight* is both what the density needs and
analytically simpler than the polynomial antiderivative: the substitution
`x = cos θ` turns each term into a plain cosine integral,

    ∫ₐᵇ Tₖ(x)/√(1-x²) dx = −∫_{acos a}^{acos b} cos(kθ) dθ ,

giving the closed forms

    I₀ = acos(a) − acos(b),
    Iₖ = ( sin(k·acos(a)) − sin(k·acos(b)) ) / k     (k ≥ 1),

and the result is `Σₖ cₖ Iₖ`. This is exact (no quadrature error) up to the KPM
moment truncation already baked into `coefficients`.

# Arguments
- `coefficients::AbstractVector{<:Real}`: Chebyshev coefficients `c₀ … c_{K}`
  (already kernel-damped and including the `k==0 ? 1 : 2` factor if reconstructing
  a KPM density — see `resolve_bounds`).
- `a`, `b`: integration bounds, `-1 ≤ a ≤ b ≤ 1`.

# Returns
- The weighted definite integral (a `Float64`).

# Notes
POLFED-agnostic numerical utility. As a sanity check, integrating over the full
interval isolates the `k = 0` term: `I₀ = acos(-1) − acos(1) = π` and `Iₖ = 0` for
`k ≥ 1`, so `∫_{-1}^{1} = π·c₀`.
"""
function integrate_chebyshev_weighted(coefficients::AbstractVector{<:Real}, a::Real, b::Real)
    θa = acos(clamp(float(a), -1.0, 1.0))
    θb = acos(clamp(float(b), -1.0, 1.0))

    # k = 0 term: ∫ 1/√(1-x²) dx = acos(a) - acos(b)
    acc = length(coefficients) == 0 ? 0.0 : coefficients[1] * (θa - θb)

    @inbounds for k in 1:length(coefficients)-1
        acc += coefficients[k+1] * (sin(k * θa) - sin(k * θb)) / k
    end

    return acc
end
