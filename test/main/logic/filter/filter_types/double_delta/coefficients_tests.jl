# DoubleDeltaFilter coefficient normalization and symmetry.

using Test
using Polfed

@testset "DoubleDeltaFilter coefficients" begin
    @testset "symmetric targets and equal orders produce an odd polynomial" begin
        target1 = -0.3
        target2 = 0.3
        half_width = 1e-2
        interval1 = Polfed.Interval(target1 - half_width, target1 + half_width)
        interval2 = Polfed.Interval(target2 - half_width, target2 + half_width)
        filter = Polfed.DoubleDeltaFilter(
            interval1,
            interval2;
            order1 = 25,
            order2 = 25,
            order_safety_factor = 1,
        )
        polynomial = Polfed._scalar_filter(filter)

        @test polynomial(target1) ≈ 1.0 atol = 1e-10
        @test polynomial(target2) ≈ -1.0 atol = 1e-10

        for x in range(-1, 1; length = 100)
            @test abs(polynomial(x)) ≈ abs(polynomial(-x)) atol = 1e-10
            @test polynomial(x) ≈ -polynomial(-x) atol = 1e-10
        end
    end

    @testset "each peak stays normalized for unequal orders and targets" begin
        half_width = 1e-2
        for order1 in 25:5:35,
            order2 in 45:50,
            target1 in -0.7:0.1:-0.5,
            target2 in 0.35:0.1:0.45

            interval1 = Polfed.Interval(target1 - half_width, target1 + half_width)
            interval2 = Polfed.Interval(target2 - half_width, target2 + half_width)
            filter = Polfed.DoubleDeltaFilter(
                interval1,
                interval2;
                order1,
                order2,
                order_safety_factor = 1,
            )
            polynomial = Polfed._scalar_filter(filter)

            @test polynomial(target1) ≈ 1.0 atol = 1e-10
            @test polynomial(target2) ≈ -1.0 atol = 1e-10
        end
    end
end
