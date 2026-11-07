using Test
using Polfed

function gaussian_dos(peak, width)
    moments = [1.0]
    coefficients = [1.0]
    ρ = x -> exp(-((x - peak)^2) / width^2)
    return Polfed.DoSResult(moments, coefficients, ρ)
end

@testset "resolve_target" begin
    peak = 0.4
    width = 0.1
    dos = gaussian_dos(peak, width)

    # Emin = -3, Emax = 5 gives a = 4, b = 1, so E = 1 maps to λ̃ = 0.
    Emin = -3.
    Emax = 5.
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2
    howmany = 5



    @testset "returns Target for valid target specs" begin
        target_specs = [
            Polfed.TargetMaxDoS(),
            Polfed.TargetMiddle(),
            Polfed.TargetOffset(0.5),
            Polfed.TargetOffset(-0.5),
            Polfed.TargetAbsolute(0.5),
            Polfed.TargetRescaled(0.5),
        ]

        for target_spec in target_specs
            target = Polfed.resolve_target(target_spec, howmany, dos, a, b)

            @test target isa Polfed.Target
            @test target.howmany == howmany
            @test target.rescaled_energy_position == :center
            @test -1 <= target.rescaled_energy <= 1
        end
    end

    @testset "absolute energy uses only rescaling parameters" begin
        target = Polfed.resolve_target(Polfed.TargetAbsolute(1.0), howmany, dos, a, b)
        @test target.rescaled_energy == 0.0

        target = Polfed.resolve_target(Polfed.TargetAbsolute(Emax), howmany, dos, a, b)
        @test target.rescaled_energy == 1.0

        target = Polfed.resolve_target(Polfed.TargetAbsolute(Emin), howmany, dos, a, b)
        @test target.rescaled_energy == -1.0

        @test_throws ArgumentError Polfed.resolve_target(Polfed.TargetAbsolute(9.0), howmany, dos, a, b)
    end

    @testset "target at a spectrum edge snaps within tolerance" begin
        # Emin/Emax are Ritz estimates converging from INSIDE the spectrum, so targeting an
        # extremal energy exactly can rescale to ∓(1 + ε) at rounding scale — snapped, not rejected.
        just_below = Polfed.resolve_target(Polfed.TargetAbsolute(Emin - a * 1e-12), howmany, dos, a, b)
        @test just_below.rescaled_energy == -1.0
        just_above = Polfed.resolve_target(Polfed.TargetAbsolute(Emax + a * 1e-12), howmany, dos, a, b)
        @test just_above.rescaled_energy == 1.0
        # beyond the snap tolerance the target is genuinely outside → still an error
        @test_throws ArgumentError Polfed.resolve_target(Polfed.TargetAbsolute(Emin - a * 1e-6), howmany, dos, a, b)
    end

    @testset "maxdos finds Gaussian peak" begin
        target = Polfed.resolve_target(Polfed.TargetMaxDoS(), howmany, dos, a, b)
        @test target.rescaled_energy ≈ peak atol=1e-5
    end

    @testset "maxdos refines around neighboring grid points" begin
        narrow_peak = 0.373
        narrow_dos = gaussian_dos(narrow_peak, 0.02)

        center = Polfed._resolve_center(
            Polfed.TargetMaxDoS(),
            narrow_dos,
            a,
            b;
            npts = 100,
            refinements = 2,
        )

        @test center ≈ narrow_peak atol=1e-5
    end

    @testset "target equivalences and offset interpolation" begin 
        maxdos = Polfed.resolve_target(Polfed.TargetMaxDoS(), howmany, dos, a, b)

        middle = Polfed.resolve_target(Polfed.TargetMiddle(), howmany, dos, a, b)

        unresc_min = Polfed.resolve_target(Polfed.TargetAbsolute(Emin), howmany, dos, a, b)
        unresc_max = Polfed.resolve_target(Polfed.TargetAbsolute(Emax), howmany, dos, a, b)

        resc0 = Polfed.resolve_target(Polfed.TargetRescaled(0.0), howmany, dos, a, b)
        resc1p = Polfed.resolve_target(Polfed.TargetRescaled(1.0), howmany, dos, a, b)
        resc1m = Polfed.resolve_target(Polfed.TargetRescaled(-1.0), howmany, dos, a, b)

        offset_0p = Polfed.resolve_target(Polfed.TargetOffset(0.0), howmany, dos, a, b)
        offset_0m = Polfed.resolve_target(Polfed.TargetOffset(-0.0), howmany, dos, a, b)
        offset_1p = Polfed.resolve_target(Polfed.TargetOffset(1.0), howmany, dos, a, b)
        offset_1m = Polfed.resolve_target(Polfed.TargetOffset(-1.0), howmany, dos, a, b)



        @test middle == resc0
        @test unresc_min == resc1m
        @test unresc_max == resc1p
        @test offset_0m == offset_0p
        @test offset_0p == maxdos
        @test offset_0m == maxdos
        @test offset_1p == resc1p
        @test offset_1m == resc1m


    end

end
