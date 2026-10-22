using Test
using Polfed

@testset "polynomial_order filter interface" begin
    interval = Polfed.Interval(-0.2, 0.1)

    up = Polfed.DeltaUpFilter(interval; order=7)
    down = Polfed.DeltaDownFilter(interval; order=11)
    @test Polfed.polynomial_order(up) === 7
    @test Polfed.polynomial_order(down) === 11

    left = Polfed.Interval(-0.8, -0.7)
    right = Polfed.Interval(0.2, 0.3)
    double = Polfed.DoubleDeltaFilter(left, right; order1=9, order2=14)
    @test Polfed.polynomial_order(double) === 14

    # The combined degree follows whichever lobe has the larger order.
    double.order1 = 17
    @test Polfed.polynomial_order(double) === 17

    # Generic scalar construction consumes the same interface and therefore
    # builds the complete combined coefficient sequence.
    expansion = Polfed._scalar_filter(double)
    @test expansion.order === Polfed.polynomial_order(double)
end
