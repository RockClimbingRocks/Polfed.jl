using Test
using UniPolfed

K = UniPolfed.Numerics.KPM

@testset "kernels" begin

    @testset "get_kernel lookup" begin
        @test K.get_kernel(:Jackson) === K.Jackson
        @test K.get_kernel(:Dirichlet) === K.Dirichlet
        @test_throws ArgumentError K.get_kernel(:Bogus)
    end

    @testset "DC moment is preserved (g(0,N) = 1)" begin
        N = 64
        @test K.Jackson(0, N) ≈ 1
        @test K.Dirichlet(0, N) ≈ 1
        @test K.Fejer(0, N) ≈ 1
        @test K.LanczosK(0, N) ≈ 1
        @test K.WangZunger(0, N) ≈ 1
        @test K.Lorentz(0, N) ≈ 1
    end

    @testset "explicit values" begin
        N = 10
        @test K.Dirichlet(7, N) == 1
        @test K.Fejer(3, N) ≈ 1 - 3/10
        @test K.Fejer(N, N) ≈ 0
        # Jackson weights are a valid damping profile: within [0,1] and decreasing
        # (the n=N weight is ~0 and may be a tiny negative from rounding)
        js = [K.Jackson(n, N) for n in 0:N]
        @test all(-1e-12 .≤ js .≤ 1 + 1e-12)
        @test issorted(js; rev=true)

        # LanczosK at n ≠ 0: sin(π)/π = 0 ⇒ weight 0 at n = N; interior in (0,1)
        @test K.LanczosK(N, N) ≈ 0 atol = 1e-12
        @test 0 < K.LanczosK(5, N) < 1
    end
end
