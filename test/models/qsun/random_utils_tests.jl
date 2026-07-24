using Test
using UniPolfed
using LinearAlgebra
using Random

Q = UniPolfed.Models.QSun

@testset "random_utils" begin

    @testset "uniform_array — range, length, reproducibility" begin
        a, b, n = -0.3, 1.7, 500
        x = Q.uniform_array(n, a, b; rng = MersenneTwister(7))
        @test length(x) == n
        @test all(a .<= x .< b)
        @test x == Q.uniform_array(n, a, b; rng = MersenneTwister(7))   # seeded ⇒ identical
        @test x != Q.uniform_array(n, a, b; rng = MersenneTwister(8))
    end

    @testset "goe_matrix — symmetric, sized, reproducible" begin
        n = 12
        M = Q.goe_matrix(n; rng = MersenneTwister(3))
        @test size(M) == (n, n)
        @test M ≈ M'                                  # GOE ⇒ real symmetric
        @test issymmetric(M)
        @test M == Q.goe_matrix(n; rng = MersenneTwister(3))
    end

    @testset "random_neighbors — within grain, length, reproducible" begin
        L_loc, L_grain = 30, 5
        nb = Q.random_neighbors(L_loc, L_grain; rng = MersenneTwister(2))
        @test length(nb) == L_loc
        @test all(0 .<= nb .<= L_grain - 1)
        @test nb == Q.random_neighbors(L_loc, L_grain; rng = MersenneTwister(2))
    end

    @testset "long_range_couplings — exact profile at ζ=0" begin
        L_loc, α = 6, 0.5
        c = Q.long_range_couplings(L_loc, α, 0.0; rng = MersenneTwister(1))
        # ζ = 0 removes the randomness ⇒ c[j] = α^(j-1) exactly.
        @test c ≈ α .^ (0:(L_loc - 1))
        @test c[1] == 1
    end

    @testset "long_range_couplings — randomized exponent stays in band" begin
        L_loc, α, ζ = 8, 0.4, 0.2
        c = Q.long_range_couplings(L_loc, α, ζ; rng = MersenneTwister(5))
        @test length(c) == L_loc
        @test c[1] == 1                               # first ray is always unit, no randomness
        # 0 < α < 1 ⇒ α^u decreasing in u; u_j ∈ [(j-1)-ζ, (j-1)+ζ]
        for j in 2:L_loc
            lo = α^((j - 1) + ζ)
            hi = α^((j - 1) - ζ)
            @test lo <= c[j] <= hi
        end
    end

    @testset "long_range_couplings — α ≤ 0 disables couplings" begin
        @test Q.long_range_couplings(5, 0.0, 0.2; rng = MersenneTwister(1)) == zeros(5)
        @test Q.long_range_couplings(5, -1.0, 0.2; rng = MersenneTwister(1)) == zeros(5)
    end
end
