using Test
using UniPolfed

@testset "polynomial_order filter interface" begin
    interval = UniPolfed.Interval(-0.2, 0.1)

    up = UniPolfed.DeltaUpFilter(interval; order=7)
    down = UniPolfed.DeltaDownFilter(interval; order=11)
    @test UniPolfed.polynomial_order(up) === 7
    @test UniPolfed.polynomial_order(down) === 11

    left = UniPolfed.Interval(-0.8, -0.7)
    right = UniPolfed.Interval(0.2, 0.3)
    double = UniPolfed.DoubleDeltaFilter(left, right; order1=9, order2=14)
    @test UniPolfed.polynomial_order(double) === 14

    # The combined degree follows whichever lobe has the larger order.
    double.order1 = 17
    @test UniPolfed.polynomial_order(double) === 17

    # Generic scalar construction consumes the same interface and therefore
    # builds the complete combined coefficient sequence.
    expansion = UniPolfed._scalar_filter(double)
    @test expansion.order === UniPolfed.polynomial_order(double)
end
