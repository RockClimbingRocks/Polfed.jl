using Test
using UniPolfed

function _test_valid_delta_up(filter)
    P = UniPolfed._scalar_filter(filter)
    interval = filter.interval
    passband = filter.passband

    @test interval.left < filter.target < interval.right
    @test passband.left < filter.target < passband.right
    @test passband.left <= interval.left
    @test passband.right >= interval.right
    @test P(interval.left) >= filter.cutoff
    @test P(interval.right) >= filter.cutoff
    @test P(passband.left) ≈ filter.cutoff atol = 1e-8
    @test P(passband.right) ≈ filter.cutoff atol = 1e-8
    @test P(filter.target) ≈ 1.0 atol = 1e-8
    @test UniPolfed.validate_filter(filter)
end

@testset "DeltaUpFilter construction" begin
    @testset "resolved order and passband" begin
        interval = UniPolfed.Interval(-0.1, 0.1)
        filter = UniPolfed.build_filter(
            UniPolfed.DeltaUpFilter,
            interval;
            cutoff = 0.17,
            order_safety_factor = 0.97,
        )

        @test filter.order == 25
        @test filter.target == 0.0
        @test filter.interval == interval
        _test_valid_delta_up(filter)
    end

    @testset "symmetric windows, half-width=$half_width" for half_width in
        (0.6, 0.2, 0.05, 1e-2, 1e-3)
        interval = UniPolfed.Interval(-half_width, half_width)
        filter = UniPolfed.build_filter(
            UniPolfed.DeltaUpFilter,
            interval;
            cutoff = 0.17,
            order_safety_factor = 0.97,
        )
        P = UniPolfed._scalar_filter(filter)

        _test_valid_delta_up(filter)
        @test P(interval.left) ≈ P(interval.right) atol = 1e-10
    end

    @testset "off-center windows" begin
        for (target, half_width) in
            ((0.2, 0.05), (0.4, 1e-2), (0.7, 1e-3), (-0.35, 0.02))
            interval = UniPolfed.Interval(target - half_width, target + half_width)
            filter = UniPolfed.build_filter(
                UniPolfed.DeltaUpFilter,
                interval;
                cutoff = 0.17,
                order_safety_factor = 0.97,
            )
            _test_valid_delta_up(filter)
        end
    end

    @testset "a caller-supplied order is honored" begin
        filter = UniPolfed.build_filter(
            UniPolfed.DeltaUpFilter,
            UniPolfed.Interval(-0.1, 0.1);
            cutoff = 0.17,
            order = 17,
        )

        @test filter.order == 17
        _test_valid_delta_up(filter)
    end

    @testset "very wide interior windows fail explicitly when no delta validates" begin
        supported = UniPolfed.resolve_filter(UniPolfed.Interval(-0.8, 0.8))
        @test supported isa UniPolfed.DeltaUpFilter
        @test UniPolfed.validate_filter(supported)

        error = try
            UniPolfed.resolve_filter(UniPolfed.Interval(-0.9, 0.9))
            nothing
        catch caught
            caught
        end
        @test error isa ErrorException
        @test occursin("no filter validated", sprint(showerror, error))
    end
end
