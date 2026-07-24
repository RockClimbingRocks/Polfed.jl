# DeltaUpFilter: construction, the interface methods, and the order/passband/validation logic.
# Ground truth is the delta-peaked Chebyshev polynomial itself — normalized to peak at exactly +1
# at the target, dropping to `cutoff` at the passband edges.

using Test
using UniPolfed

@testset "DeltaUpFilter - building filter" begin


    @testset "Vertify exact results for interval" begin
        iv = UniPolfed.Interval(-0.1, 0.1)     
        f = UniPolfed.build_filter(
            UniPolfed.DeltaUpFilter, iv; 
            cutoff = 0.17, 
            order_safety_factor = 0.97
        )

        P = UniPolfed._scalar_filter(f)

        # Exact results obtianed numerically
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
        @test P(f.interval.left)    >= f.cutoff 
        @test P(f.interval.right)   >= f.cutoff     
        @test P(f.passband.left)    ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right)   ≈  f.cutoff atol = 1e-8
        @test P(f.target)           ≈  1.0      atol = 1e-8  
    end
    
    @testset "Vertify identities - symmetric filters, δ=$(δ)" for δ in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-2, 1e-3, 1e-4, 1e-5] 
        iv = UniPolfed.Interval(-δ, δ)
        f = UniPolfed.build_filter(
            UniPolfed.DeltaUpFilter, iv;
            cutoff = 0.17,
            order_safety_factor = 0.97
        )
        P = UniPolfed._scalar_filter(f)

        @test f.interval.left       <  f.target             < f.interval.right
        @test f.passband.left       <  f.target             < f.passband.right
        @test f.passband.left       <= f.interval.left
        @test f.passband.right      >= f.interval.right
        @test P(f.interval.left)    >= f.cutoff 
        @test P(f.interval.right)   >= f.cutoff
        @test P(f.passband.left)    ≈  f.cutoff atol = 1e-8
        @test P(f.passband.right)   ≈  f.cutoff atol = 1e-8
        @test P(f.target)           ≈  1.0      atol = 1e-8

        # Properties of symmetric filters
        @test P(f.interval.right)   ≈  P(f.interval.left) 
    end

    @testset "Verify identities - non-symmetric filter, target=$(target), δ=$(δ)" for δ in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], target in [1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6, 0.7, 0.8]
        iv = UniPolfed.Interval(target - δ, target + δ)
        f = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, iv;
            cutoff = 0.17,
            order_safety_factor = 0.97,
        )
        P = UniPolfed._scalar_filter(f)

        # if δ==0.1 && target==0.8
        #     println("-----------")
        #     display(f)
        # end

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

    @testset "Test predetermined order..." begin
        @test false
    end

    @testset "Dont forget to test δ=0.8 and above... What to do with it?" begin
        @test false
    end

end
