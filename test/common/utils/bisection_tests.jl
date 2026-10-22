using Test
using Polfed

@testset "bisection" begin
    @testset "roots of known functions" begin
        @test Polfed.bisection(cos, 0.0, pi) ≈ pi / 2 atol = 1e-7
        @test Polfed.bisection(x -> x^2 - 2, 0.0, 2.0) ≈ sqrt(2) atol = 1e-7
        @test Polfed.bisection(x -> x^3 - x - 2, 1.0, 2.0) ≈ 1.5213797 atol = 1e-6
        # linear root exactly at an endpoint-bracketed interior
        @test Polfed.bisection(x -> 2x - 1, -1.0, 1.0) ≈ 0.5 atol = 1e-7
    end

    @testset "tolerance controls the half-width" begin
        r = Polfed.bisection(x -> x - 0.3, 0.0, 1.0; tol = 1e-3)
        @test abs(r - 0.3) <= 1e-3
    end

    @testset "no sign change throws" begin
        @test_throws ErrorException Polfed.bisection(x -> x^2 + 1, -1.0, 1.0)  # always positive
        @test_throws ErrorException Polfed.bisection(x -> x + 5, 0.0, 1.0)     # same sign both ends
    end
end
