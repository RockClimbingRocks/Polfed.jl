using Test
using UniPolfed
using LinearAlgebra

K = UniPolfed.Numerics.KPM

chebT(x, k) = cos(k * acos(x))

# Exact (trace) moments of a diagonal H with spectrum λ (all in (-1,1)):
# μₖ = Σⱼ Tₖ(λⱼ).
exact_moments(λ, N) = [sum(chebT(λj, k) for λj in λ) for k in 0:N-1]

# ∫_{-1}^{1} ρ(x) dx  via  x = cosθ  ⇒  ∫₀^π ρ(cosθ)·sinθ dθ  (midpoint rule;
# the substitution removes the 1/√(1-x²) endpoint singularity).
function integrate_density(ρ; M = 4000)
    s = 0.0
    for m in 1:M
        θ = (m - 0.5) * π / M
        s += ρ(cos(θ)) * sin(θ)
    end
    return s * π / M
end

@testset "density of states" begin

    @testset "∫ρ over [-1,1] equals μ₀ (total weight)" begin
        λ = [0.3, -0.7, 0.5, -0.1, 0.85]
        N = 40
        μ = exact_moments(λ, N)
        ρ = K.kpm_density(μ; kernel = :Jackson)
        # ∫ρ = g(0,N)·μ₀ = μ₀ = length(λ) (since Jackson(0,N)=1, T₀≡1)
        @test integrate_density(ρ) ≈ μ[1] rtol = 1e-3
        @test μ[1] ≈ length(λ)
    end


    @testset "Dirichlet reproduces the raw Chebyshev series" begin
        λ = [0.2, -0.4, 0.6]
        N = 24
        μ = exact_moments(λ, N)
        ρ = K.kpm_density(μ; kernel = :Dirichlet)
        raw(x) = (μ[1] + 2 * sum(μ[k + 1] * chebT(x, k) for k in 1:N-1)) /
                 (π * sqrt(1 - x^2))
        for x in (-0.8, -0.3, 0.0, 0.5, 0.9)
            @test ρ(x) ≈ raw(x)
        end
    end


    @testset "reconstructed density is finite on the interior" begin
        λ = [0.3, -0.7, 0.5]
        ρ = K.kpm_density(exact_moments(λ, 32))
        @test all(isfinite, ρ.(range(-0.95, 0.95; length = 50)))
    end

    @testset "density accepts moments from a complex Hermitian spectrum" begin
        H = ComplexF64[
            0.20       0.10 + 0.20im  0.05 - 0.03im
            0.10 - 0.20im  -0.30      0.07 + 0.04im
            0.05 + 0.03im   0.07 - 0.04im  0.40
        ]
        λ = eigvals(Hermitian(H))
        @test all(abs.(λ) .< 1)

        μ = exact_moments(λ, 40)
        ρ = K.kpm_density(μ; kernel = :Jackson)

        @test all(isfinite, ρ.(range(-0.95, 0.95; length = 50)))
        @test integrate_density(ρ) ≈ length(λ) rtol = 1e-3
    end
    

    @testset "ρ(x) is type-stable" begin
        ρ = K.kpm_density(exact_moments([0.3, -0.7, 0.5], 32))
        # regression guard: family is a ClenshawExpansion type parameter, so the
        # scalar evaluation inside ρ infers concretely (was ::Any before).
        @test @inferred(ρ(0.3)) isa Float64
    end
end
