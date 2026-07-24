# SpectralTransformReport: target-strategy strings, the constructor's rescaling/cost math, and
# section rendering.

using LinearAlgebra
using UniPolfed: MappingConfig, NoParallel, build_mapping_plan
import UniPolfed: SpectralTransformReport, format_target_strategy, render_transform,
              Interval, DeltaUpFilter, TargetMiddle, TargetMaxDoS, TargetOffset,
              TargetAbsolute, TargetRescaled, Formatter

@testset "SpectralTransformReport" begin
    @testset "format_target_strategy" begin
        @test format_target_strategy(TargetMaxDoS()) == ":maxdos"
        @test format_target_strategy(TargetMiddle()) == ":middle"
        @test occursin("(:offset,",      format_target_strategy(TargetOffset(0.25)))
        @test occursin("(:unrescaled,",  format_target_strategy(TargetAbsolute(0.13)))
        @test occursin("(:rescaled,",    format_target_strategy(TargetRescaled(0.05)))
    end

    @testset "constructor rescaling + cost math" begin
        D = 16
        A = Diagonal(range(-1.0, 1.0; length = D))
        f!(Y, X) = mul!(Y, A, X)
        mp = build_mapping_plan(MappingConfig(Emin = -2.0, Emax = 2.0,
                                              parallel_strategy = NoParallel()),
                                f!, randn(D, 2))  # a=2, b=0
        interval = Interval(-0.1, 0.1)
        filter = DeltaUpFilter(Interval(0.0, 0.1); order = 100)   # target = midpoint = 0.05, fixed order

        st = SpectralTransformReport(interval, filter, mp, TargetMiddle(), 2, 50, 6)
        @test st.target_rescaled ≈ 0.05
        @test st.target_energy ≈ mp.a * 0.05 + mp.b          # E = spread·x̃ + center
        @test st.energy_left  ≈ mp.a * (-0.1) + mp.b
        @test st.energy_right ≈ mp.a * ( 0.1) + mp.b
        @test st.order == 100
        @test st.filter_name == "DeltaUpFilter"
        @test st.polynomial_type == "Chebyshev"
        @test st.matvecs == 100 * 2 * 50 + 6                 # K·blocksize·iters + howmany
        @test st.target_strategy == ":middle"
    end

    @testset "render_transform content" begin
        st = SpectralTransformReport(0.13, 0.0166, ":maxdos", -0.0083, 0.0117,
                                     0.07, 0.19, "DeltaUpFilter", "Chebyshev", 214, 0.17, 471_800)
        s = sprint(io -> render_transform(io, Formatter(false), st))
        @test occursin("Spectral Transformation", s)
        @test occursin(":maxdos", s)
        @test occursin("order K = 214", s)
        @test occursin("DeltaUpFilter", s)
        @test occursin("cutoff 0.17", s)                     # the resolved suppression cutoff
        @test occursin("mat-vec muls: 471_800", s)           # underscore-grouped cost
        @test !occursin("safety", s)                         # dropped, replaced by cutoff
        @test !occursin("coefs", s)
        @test !occursin("\e[", s)
    end

    @testset "a cutoff-less filter (NaN) omits the cutoff clause" begin
        st = SpectralTransformReport(0.13, 0.0166, ":maxdos", -0.0083, 0.0117,
                                     0.07, 0.19, "EdgeFilter", "Chebyshev", 3, NaN, 471_800)
        s = sprint(io -> render_transform(io, Formatter(false), st))
        @test occursin("order K = 3", s)
        @test !occursin("cutoff", s)                         # NaN ⇒ no cutoff shown
        @test !occursin("NaN", s)
    end
end
