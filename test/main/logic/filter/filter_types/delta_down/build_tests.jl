
using Test
using Polfed

@testset "DeltaDownFilter construction" begin
    @testset "exact interval contract" begin
        iv = Polfed.Interval(-0.1, 0.1)
        f = Polfed.build_filter(
            Polfed.DeltaDownFilter, iv;
            cutoff = -0.17,
            order_safety_factor = 0.97,
        )

        P = Polfed._scalar_filter(f)

        # Exact results obtained numerically.
        @test f.order == 25
        @test f.target ≈ (iv.left + iv.right)/2 atol = 1e-8
        @test f.interval == iv
        @test f.passband.left  ≈ -0.10664465656736866
        @test f.passband.right ≈ +0.10664465656736866

        # Identities and inequalities that must hold for a valid filter
        @test f.interval.left       <  f.target             < f.interval.right
        @test f.passband.left       <  f.target             < f.passband.right
        @test f.passband.left       <= f.interval.left
        @test f.passband.right      >= f.interval.right
        @test P(f.interval.left)    <= f.cutoff
        @test P(f.interval.right)   <= f.cutoff
        @test P(f.passband.left)    ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right)   ≈  f.cutoff atol = 1e-8
        @test P(f.target)           ≈  -1.0      atol = 1e-8
    end

    @testset "symmetric filter identities, δ=$(δ)" for δ in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
        iv = Polfed.Interval(-δ, δ)
        f = Polfed.build_filter(
            Polfed.DeltaDownFilter, iv;
            cutoff = -0.17,
            order_safety_factor = 0.97,
        )
        P = Polfed._scalar_filter(f)

        @test f.interval.left       <  f.target             < f.interval.right
        @test f.passband.left       <  f.target             < f.passband.right
        @test f.passband.left       <= f.interval.left
        @test f.passband.right      >= f.interval.right
        @test P(f.interval.left)    <= f.cutoff
        @test P(f.interval.right)   <= f.cutoff
        @test P(f.passband.left)    ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right)   ≈  f.cutoff atol = 1e-8
        @test P(f.target)           ≈  -1.0      atol = 1e-8

        # Properties of symmetric filters
        @test P(f.interval.right)   ≈  P(f.interval.left)
    end

    @testset "Verify identities - non-symmetric filter, target=$(target), δ=$(δ)" for δ in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], target in [1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6, 0.7, 0.8]
        iv = Polfed.Interval(target - δ, target + δ)
        f = Polfed.build_filter(Polfed.DeltaDownFilter, iv;
            cutoff = -0.17,
            order_safety_factor = 0.97,
        )
        P = Polfed._scalar_filter(f)


        @test f.interval.left     <  f.target < f.interval.right
        @test f.passband.left     <  f.target < f.passband.right
        @test f.passband.left     <= f.interval.left
        @test f.passband.right    >= f.interval.right
        @test P(f.interval.left)  <= f.cutoff
        @test P(f.interval.right) <= f.cutoff
        @test P(f.passband.left)  ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right) ≈  f.cutoff atol = 1e-8
        @test P(f.target)         ≈  -1.0 atol = 1e-8
    end
end
