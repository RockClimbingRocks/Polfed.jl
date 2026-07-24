using Test
using UniPolfed

function gaussian_dos(peak, width)
    moments = [1.0]
    coefficients = [1.0]
    ρ = x -> exp(-((x - peak)^2) / width^2)
    return UniPolfed.DoSResult(moments, coefficients, ρ)
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
            UniPolfed.TargetMaxDoS(),
            UniPolfed.TargetMiddle(),
            UniPolfed.TargetOffset(0.5),
            UniPolfed.TargetOffset(-0.5),
            UniPolfed.TargetAbsolute(0.5),
            UniPolfed.TargetRescaled(0.5),
        ]

        for target_spec in target_specs
            target = UniPolfed.resolve_target(target_spec, howmany, dos, a, b)

            @test target isa UniPolfed.Target
            @test target.howmany == howmany
            @test target.rescaled_energy_position == :center
            @test -1 <= target.rescaled_energy <= 1
        end
    end

    @testset "absolute energy uses only rescaling parameters" begin
        target = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(1.0), howmany, dos, a, b)
        @test target.rescaled_energy == 0.0

        target = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emax), howmany, dos, a, b)
        @test target.rescaled_energy == 1.0

        target = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emin), howmany, dos, a, b)
        @test target.rescaled_energy == -1.0

        @test_throws ArgumentError UniPolfed.resolve_target(UniPolfed.TargetAbsolute(9.0), howmany, dos, a, b)
    end

    @testset "target at a spectrum edge snaps within tolerance" begin
        # Emin/Emax are Ritz estimates converging from INSIDE the spectrum, so targeting an
        # extremal energy exactly can rescale to ∓(1 + ε) at rounding scale — snapped, not rejected.
        just_below = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emin - a * 1e-12), howmany, dos, a, b)
        @test just_below.rescaled_energy == -1.0
        just_above = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emax + a * 1e-12), howmany, dos, a, b)
        @test just_above.rescaled_energy == 1.0
        # beyond the snap tolerance the target is genuinely outside → still an error
        @test_throws ArgumentError UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emin - a * 1e-6), howmany, dos, a, b)
    end

    @testset "maxdos finds Gaussian peak" begin
        target = UniPolfed.resolve_target(UniPolfed.TargetMaxDoS(), howmany, dos, a, b)
        @test target.rescaled_energy ≈ peak atol=1e-5
    end

    @testset "maxdos refines around neighboring grid points" begin
        narrow_peak = 0.373
        narrow_dos = gaussian_dos(narrow_peak, 0.02)

        center = UniPolfed._resolve_center(
            UniPolfed.TargetMaxDoS(),
            narrow_dos,
            a,
            b;
            npts = 100,
            refinements = 2,
        )

        @test center ≈ narrow_peak atol=1e-5
    end

    @testset "target equivalences and offset interpolation" begin 
        maxdos = UniPolfed.resolve_target(UniPolfed.TargetMaxDoS(), howmany, dos, a, b)

        middle = UniPolfed.resolve_target(UniPolfed.TargetMiddle(), howmany, dos, a, b)

        unresc_min = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emin), howmany, dos, a, b)
        unresc_max = UniPolfed.resolve_target(UniPolfed.TargetAbsolute(Emax), howmany, dos, a, b)

        resc0 = UniPolfed.resolve_target(UniPolfed.TargetRescaled(0.0), howmany, dos, a, b)
        resc1p = UniPolfed.resolve_target(UniPolfed.TargetRescaled(1.0), howmany, dos, a, b)
        resc1m = UniPolfed.resolve_target(UniPolfed.TargetRescaled(-1.0), howmany, dos, a, b)

        offset_0p = UniPolfed.resolve_target(UniPolfed.TargetOffset(0.0), howmany, dos, a, b)
        offset_0m = UniPolfed.resolve_target(UniPolfed.TargetOffset(-0.0), howmany, dos, a, b)
        offset_1p = UniPolfed.resolve_target(UniPolfed.TargetOffset(1.0), howmany, dos, a, b)
        offset_1m = UniPolfed.resolve_target(UniPolfed.TargetOffset(-1.0), howmany, dos, a, b)



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
