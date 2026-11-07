using Test
using Polfed

# Ground truth: ∫ₐᵇ Tₖ(x)/√(1-x²) dx has the closed forms
#   I₀ = acos(a) - acos(b),  Iₖ = (sin(k·acos a) - sin(k·acos b))/k.
# We check these against independent hand calculations and a fine numeric integral.

# reference numeric integral of (Σ cₖ Tₖ(x))/√(1-x²) via substitution x = cos θ,
# which removes the singularity: ∫ (Σ cₖ cos(kθ)) dθ over θ ∈ [acos b, acos a].
function ref_weighted_integral(coeffs, a, b; npts = 2_000_000)
    θs = range(acos(b), acos(a); length = npts)   # increasing θ
    dθ = step(θs)
    s = 0.0
    for θ in θs
        val = 0.0
        for (k, c) in enumerate(coeffs)
            val += c * cos((k - 1) * θ)
        end
        s += val
    end
    return s * dθ
end

@testset "integrate_chebyshev_weighted" begin
    @testset "single-term ground truth over [-1,1]" begin
        # ∫ T0/√ = π ; ∫ Tₖ/√ = 0 for k ≥ 1
        @test Polfed.integrate_chebyshev_weighted([1.0], -1.0, 1.0) ≈ pi atol = 1e-12
        for k in 1:6
            c = zeros(k + 1); c[k+1] = 1.0
            @test Polfed.integrate_chebyshev_weighted(c, -1.0, 1.0) ≈ 0 atol = 1e-12
        end
    end

    @testset "partial-interval closed forms" begin
        # ∫₀¹ T0/√ = acos(0) - acos(1) = π/2
        @test Polfed.integrate_chebyshev_weighted([1.0], 0.0, 1.0) ≈ pi / 2 atol = 1e-12
        # ∫₀¹ T1/√ = ∫₀¹ x/√(1-x²) dx = 1
        @test Polfed.integrate_chebyshev_weighted([0.0, 1.0], 0.0, 1.0) ≈ 1 atol = 1e-12
        # additivity: [-1,0] + [0,1] = [-1,1]
        c = [0.7, -0.3, 0.5, 0.1]
        whole = Polfed.integrate_chebyshev_weighted(c, -1.0, 1.0)
        left = Polfed.integrate_chebyshev_weighted(c, -1.0, 0.0)
        right = Polfed.integrate_chebyshev_weighted(c, 0.0, 1.0)
        @test whole ≈ left + right atol = 1e-12
    end

    @testset "multi-term against a fine numeric integral" begin
        c = [1.0, 0.4, -0.25, 0.1, 0.05]
        for (a, b) in ((-1.0, 1.0), (-0.5, 0.8), (0.1, 0.9), (-0.9, -0.2))
            @test Polfed.integrate_chebyshev_weighted(c, a, b) ≈ ref_weighted_integral(c, a, b) atol = 1e-4
        end
    end

    @testset "empty / degenerate" begin
        @test Polfed.integrate_chebyshev_weighted(Float64[], -1.0, 1.0) == 0.0
        @test Polfed.integrate_chebyshev_weighted([1.0], 0.3, 0.3) ≈ 0 atol = 1e-12  # zero-width
    end
end
