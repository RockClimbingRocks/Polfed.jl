# DeltaUpFilter: construction, the interface methods, and the order/passband/validation logic.
# Ground truth is the delta-peaked Chebyshev polynomial itself — normalized to peak at exactly +1
# at the target, dropping to `cutoff` at the passband edges.

using Test
using UniPolfed

@testset "DoubleDeltaFilter - construction" begin

    @testset "Unbuilt filter has negative order" begin
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0.4, 0.5)
        f = UniPolfed.DoubleDeltaFilter(iv1, iv2)
        @test f.order1 < 0
        @test f.order2 < 0
    end

    @testset "Filter interval is set correctly" begin
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0.4, 0.5)
        f = UniPolfed.DoubleDeltaFilter(iv1, iv2)
        @test f.interval1 == iv1
        @test f.interval2 == iv2
    end

    @testset "Filter targets are set correctly" begin
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0.4, 0.5)
        f = UniPolfed.DoubleDeltaFilter(iv1, iv2)
        @test f.target1 == 0.0
        @test f.target2 == 0.45
    end

    @testset "Filter cutoff values are set correctly" begin
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0.4, 0.5)
        f = UniPolfed.DoubleDeltaFilter(iv1, iv2; cutoff_lower=-0.2, cutoff_upper=0.2)
        @test f.cutoff_lower == -0.2
        @test f.cutoff_upper == 0.2
    end

    @testset "Filter order safety factor is set correctly" begin
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0.4, 0.5)
        f = UniPolfed.DoubleDeltaFilter(iv1, iv2; order_safety_factor=0.95)
        @test f.order_safety_factor == 0.95
    end

    @testset "Filter targets targets both endes of the spectrum" begin
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0.4, 0.5)
        f = UniPolfed.DoubleDeltaFilter(iv1, iv2)
        @test UniPolfed.exposed_end(f) == :both
    end

    @testset "Filter throws error if intervals overlap" begin 
        iv1 = UniPolfed.Interval(-0.1, 0.1)
        iv2 = UniPolfed.Interval(0., 0.5)
        @test_throws ArgumentError UniPolfed.DoubleDeltaFilter(iv1, iv2)
    end     

    @testset "Filter throws error if interval1 is not on the left of interval2" begin
        iv1 = UniPolfed.Interval(+0.5, +0.75)
        iv2 = UniPolfed.Interval(-0.5, -0.20)
        @test_throws ArgumentError UniPolfed.DoubleDeltaFilter(iv1, iv2)
    end


end
