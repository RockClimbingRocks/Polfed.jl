# DeltaUpFilter: construction, the interface methods, and the order/passband/validation logic.
# Ground truth is the delta-peaked Chebyshev polynomial itself — normalized to peak at exactly +1
# at the target, dropping to `cutoff` at the passband edges.

using Test
using UniPolfed

@testset "DoubleDeltaFilter - coefficients" begin
    
    @testset "Testing symemtries of symmetric double filter" begin
        target1 = -0.3
        target2 = +0.3
        δ = 1e-2
        iv1 = UniPolfed.Interval(target1-δ, target1+δ)
        iv2 = UniPolfed.Interval(target2-δ, target2+δ)
        filter = UniPolfed.DoubleDeltaFilter(iv1, iv2; order1 = 25, order2 = 25, order_safety_factor = 1)
        # coefs = UniPolfed.coefficients(filter)

        P = UniPolfed._scalar_filter(filter)
        xs = range(-1, 1, length=100)

        peak1_at_1p = isapprox(P(target1), +1.0, atol = 1e-10)
        peak2_at_1m = isapprox(P(target2), -1.0, atol = 1e-10)

        is_abs_symm = true 
        is_sgn_asymm = true
        is_val_asymm = true
        for x in xs
            is_abs_symm = is_abs_symm && isapprox(abs(P(x)), abs(P(-x)), atol = 1e-10)
            is_sgn_asymm = is_sgn_asymm && sign(P(x)) != sign(P(-x))
            is_val_asymm = is_val_asymm && isapprox(P(x), -P(-x), atol = 1e-10)
        end
        

        @test peak1_at_1p
        @test peak2_at_1m
        @test is_abs_symm 
        @test is_sgn_asymm
        @test is_val_asymm
    end
        
    @testset "Testing symemtries of symmetric double filter" begin 

        for order1 in 25:5:35, order2 in 45:1:50, target1 in -0.7:0.1:-0.5, target2 in 0.35:0.1:0.45
            target1 = -0.3
            target2 = +0.3
            δ = 1e-2
            iv1 = UniPolfed.Interval(target1-δ, target1+δ)
            iv2 = UniPolfed.Interval(target2-δ, target2+δ)
            filter = UniPolfed.DoubleDeltaFilter(iv1, iv2; order1 = order1, order2 = order2, order_safety_factor = 1)

            P = UniPolfed._scalar_filter(filter)
            xs = range(-1, 1, length=100)

            peak1_at_1p = isapprox(P(target1), +1.0, atol = 1e-10)
            peak2_at_1m = isapprox(P(target2), -1.0, atol = 1e-10)

            @test peak1_at_1p
            @test peak2_at_1m
        end

    end

end
