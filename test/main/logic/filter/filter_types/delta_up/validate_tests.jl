# DeltaUpFilter: construction, the interface methods, and the order/passband/validation logic.
# Ground truth is the delta-peaked Chebyshev polynomial itself — normalized to peak at exactly +1
# at the target, dropping to `cutoff` at the passband edges.

using Test
using UniPolfed

@testset "DeltaUpFilter - validate filter" begin   

    @testset "Validate good symmetric filters δ∈[1e-3, 0.35]" begin 
        for δ in range(1e-3, stop=0.35, length=25)
            iv = UniPolfed.Interval(-δ, δ)
            f = UniPolfed.build_filter(
                UniPolfed.DeltaUpFilter, iv;
                cutoff = 0.17,
                order_safety_factor = 0.97
            )
            
            @test UniPolfed.validate_filter(f) == true
        end
    end

    @testset "Validate good symmetric filters, cutoffs = $(cutoff)" for cutoff in range(0.18,0.5, length=10)
        for δ in range(1e-3, stop=0.35, length=15)
            iv = UniPolfed.Interval(-δ, δ)
            f = UniPolfed.build_filter(
                UniPolfed.DeltaUpFilter, iv;
                cutoff = cutoff,
                order_safety_factor = 0.97
            )
            
            @test UniPolfed.validate_filter(f) == true
        end
    end

    @testset "Validate good non-symmetric filters order= 100, target ∈ [0, 0.98]" begin 
        order = 100
        cutoff = 0.17
        δ = 1e-3
        
        # `logrange` is Julia >= 1.11; CI also runs 1.10, so spell it out.
        for target in exp10.(range(log10(1e-3), log10(0.98), length=25))
            iv = UniPolfed.Interval(target-δ, target+δ)    
            f = UniPolfed.build_filter(
                UniPolfed.DeltaUpFilter, iv;
                order=order,
                cutoff = cutoff,
                order_safety_factor = 0.97
            )
            
            @test UniPolfed.validate_filter(f) == true
        end
    end

    @testset "Validate edge case when passband is outside [-1,1] interval." begin
        target = 0.0
        δ = 0.7
        iv = UniPolfed.Interval(target-δ, target+δ)
        f = UniPolfed.build_filter(
            UniPolfed.DeltaUpFilter, iv;
            cutoff = 0.17,
            order_safety_factor = 0.97
        )
        
        passband = UniPolfed.Interval(-1.0, 1.0)
        f.passband = passband
        @test_nowarn UniPolfed.validate_filter(f)


        passband = UniPolfed.Interval(-1.2, 0.25)
        f.passband = passband
        @test_throws ErrorException UniPolfed.validate_filter(f)

        passband = UniPolfed.Interval(-0.9, 1.2)
        f.passband = passband
        @test_throws ErrorException UniPolfed.validate_filter(f)

    end 

    @testset "Validate bad symmetric filters" begin
        target = 0.0
        cf(δ) = UniPolfed.Interval(target-δ,target+δ)

        bad_params = [
            (cf(0.1),   27),  
            (cf(0.1),   30),
            (cf(0.25),  10), 
            (cf(0.25),  11), 
            (cf(0.25),  12), 
            (cf(0.4),    4), 
            (cf(0.1),    5), 
        ]

        for param in bad_params
            iv, order = param

            f = UniPolfed.build_filter(
                UniPolfed.DeltaUpFilter, iv;
                cutoff = 0.17,
                order_safety_factor = 0.97
            )
            f.passband = iv
            
            @test UniPolfed.validate_filter(f) == false
        end
    end

    @testset "Validate bad non-symmetric filters" begin

        
        target=0.99
        cf(δ) = UniPolfed.Interval(target-δ,target+δ)
        
        ivs = [ cf(0.01), cf(0.0125), cf(0.015), cf(0.0175)]

        for (i, iv) in enumerate(ivs)
            f = UniPolfed.build_filter(
                UniPolfed.DeltaUpFilter, iv;
                cutoff = 0.17,
                order_safety_factor = 0.97
            )
            @test UniPolfed.validate_filter(f) == false
        end
    end

    # Region checks called directly, so each region's true/false branch is exercised in isolation.
    @testset "region functions" begin
        _ε    = UniPolfed.Defaults.filters.delta_up.Δε
        _rtol = UniPolfed.Defaults.filters.delta_up.leak_rtol
        Ppoly(f)    = UniPolfed._scalar_filter(f)
        pass_ok(f)  = UniPolfed.validate_passband(Ppoly(f), f, _ε, _rtol, UniPolfed.Defaults.filters.delta_up.npts_within_passband(f.order))
        tails_ok(f) = UniPolfed.validate_tails(Ppoly(f), f, _ε, _rtol, UniPolfed.Defaults.filters.delta_up.npts_on_sides_of_the_passband(f.order))

        @testset "good filter -> interior & tails clean" begin
            f = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, UniPolfed.Interval(-0.2, 0.2); cutoff = 0.17, order_safety_factor = 0.97)
            @test pass_ok(f)
            @test tails_ok(f)
            @test UniPolfed.validate_passband_within_spectrum(f)
        end

        @testset "widened passband -> interior leaks (validate_passband false)" begin
            f = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, UniPolfed.Interval(-0.2, 0.2); cutoff = 0.17, order_safety_factor = 0.97)
            f.passband = UniPolfed.Interval(f.passband.left - 0.1, f.passband.right + 0.1)   # wider than resolved
            @test pass_ok(f) == false
            @test tails_ok(f)                       # tails still clean -> isolates the interior leak
        end

        @testset "near-edge lobe -> tail leaks (validate_tails false)" begin
            f = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, UniPolfed.Interval(0.99 - 0.015, 0.99 + 0.015); cutoff = 0.17, order_safety_factor = 0.97)
            @test tails_ok(f) == false
            @test pass_ok(f)                        # interior still amplified -> isolates the tail leak
        end

        @testset "spectrum guard throws when passband exits [-1,1]" begin
            f = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, UniPolfed.Interval(-0.2, 0.2); cutoff = 0.17, order_safety_factor = 0.97)
            f.passband = UniPolfed.Interval(-1.2, 0.25)
            @test_throws ErrorException UniPolfed.validate_passband_within_spectrum(f)
            @test_throws ErrorException UniPolfed.validate_filter(f)
        end
    end

    @testset "Fit-for-request: passband must COVER filter.interval and stay interior" begin
        # A fixed order far too sharp for the requested window: the polynomial itself is clean
        # (no leakage), but the passband cannot cover the request -> validate_filter is false.
        iv = UniPolfed.Interval(-0.3, 0.3)
        f = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, iv; order = 200, cutoff = 0.17)
        @test f.passband.right - f.passband.left < iv.right - iv.left   # too narrow by construction
        @test UniPolfed.validate_filter(f) == false

        # A passband reaching a spectrum edge (forced) is not a useful delta either.
        g = UniPolfed.build_filter(UniPolfed.DeltaUpFilter, UniPolfed.Interval(-0.1, 0.1); cutoff = 0.17)
        g.passband = UniPolfed.Interval(-1.0, g.passband.right)
        @test UniPolfed.validate_filter(g) == false
    end

end
