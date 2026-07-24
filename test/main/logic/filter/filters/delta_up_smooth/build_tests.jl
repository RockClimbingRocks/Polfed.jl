# DeltaUpSmoothFilter construction: turning a window into a resolved filter (order bisected to the
# cutoff, passband recovered). The resolved order and passband differ numerically from the sharp
# DeltaUpFilter (the Jackson kernel widens the main lobe), so the ground truth here is the set of
# identities/inequalities that must hold for ANY valid delta filter — not hardcoded order/passband
# values. The damped peak still normalizes to +1 and drops to `cutoff` at the passband edges.

using Test
using UniPolfed

@testset "DeltaUpSmoothFilter - building filter" begin

    @testset "Verify identities for a single interval" begin
        iv = UniPolfed.Interval(-0.1, 0.1)
        f = UniPolfed.build_filter(
            UniPolfed.DeltaUpSmoothFilter, iv;
            cutoff = 0.17,
            order_safety_factor = 0.97,
        )
        P = UniPolfed._scalar_filter(f)

        @test f.order > 0
        @test f.target ≈ (iv.left + iv.right) / 2 atol = 1e-8
        @test f.interval == iv

        # Identities and inequalities that must hold for a valid filter
        @test f.interval.left     <  f.target < f.interval.right
        @test f.passband.left     <  f.target < f.passband.right
        @test f.passband.left     <= f.interval.left
        @test f.passband.right    >= f.interval.right
        @test P(f.interval.left)  >= f.cutoff
        @test P(f.interval.right) >= f.cutoff
        @test P(f.passband.left)  ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right) ≈  f.cutoff atol = 1e-8
        @test P(f.target)         ≈  1.0 atol = 1e-8
    end

    @testset "Verify identities - symmetric filters, δ=$(δ)" for δ in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]
        iv = UniPolfed.Interval(-δ, δ)
        f = UniPolfed.build_filter(
            UniPolfed.DeltaUpSmoothFilter, iv;
            cutoff = 0.17,
            order_safety_factor = 0.97,
        )
        P = UniPolfed._scalar_filter(f)

        @test f.interval.left     <  f.target < f.interval.right
        @test f.passband.left     <  f.target < f.passband.right
        @test f.passband.left     <= f.interval.left
        @test f.passband.right    >= f.interval.right
        @test P(f.interval.left)  >= f.cutoff
        @test P(f.interval.right) >= f.cutoff
        @test P(f.passband.left)  ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right) ≈  f.cutoff atol = 1e-8
        @test P(f.target)         ≈  1.0 atol = 1e-8

        # Properties of symmetric filters
        @test P(f.interval.right) ≈ P(f.interval.left)
    end

    @testset "Verify identities - non-symmetric filter, target=$(target), δ=$(δ)" for δ in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], target in [1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6, 0.7, 0.8]
        iv = UniPolfed.Interval(target - δ, target + δ)
        f = UniPolfed.build_filter(
            UniPolfed.DeltaUpSmoothFilter, iv;
            cutoff = 0.17,
            order_safety_factor = 0.97,
        )
        P = UniPolfed._scalar_filter(f)

        @test f.interval.left     <  f.target < f.interval.right
        @test f.passband.left     <  f.target < f.passband.right
        @test f.passband.left     <= f.interval.left
        @test f.passband.right    >= f.interval.right
        @test P(f.interval.left)  >= f.cutoff
        @test P(f.interval.right) >= f.cutoff
        @test P(f.passband.left)  ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right) ≈  f.cutoff atol = 1e-8
        @test P(f.target)         ≈  1.0 atol = 1e-8
    end

    @testset "User-fixed order is respected (no bisection)" begin
        iv = UniPolfed.Interval(-0.05, 0.05)
        f = UniPolfed.build_filter(UniPolfed.DeltaUpSmoothFilter, iv; order = 40, cutoff = 0.17)
        @test f.order == 40
    end

end
