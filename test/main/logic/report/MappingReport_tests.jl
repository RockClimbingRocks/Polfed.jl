# MappingReport: operator-string mapping, snapshot from a MappingPlan, and section rendering.

using LinearAlgebra
using UniPolfed: MappingConfig, BlockWise, build_mapping_plan
import UniPolfed: MappingReport, _operator_str, render_mapping, ColumnLayout, RowLayout, Formatter

using UniPolfed.Optimization: MappingKernel

@testset "MappingReport" begin
    @testset "_operator_str covers every optimize_mapping form" begin
        @test _operator_str(false) == "generic operator (mul! / user f!)"
        @test _operator_str(true)  == "value-grouped kernel (auto)"
        @test _operator_str(:auto) == "value-grouped kernel (auto)"
        @test _operator_str(:universal) == "value-grouped kernel :universal"
        @test _operator_str((kernel = :universal,)) == "value-grouped kernel :universal"
        @test _operator_str(MappingKernel(:universal)) ==
              "value-grouped kernel :universal_row_read"
        @test _operator_str(MappingKernel(:universal, :row_bucketed)) ==
              "value-grouped kernel :universal_row_bucketed"
    end

    @testset "snapshot from a MappingPlan" begin
        D = 16
        A = Diagonal(range(-1.0, 1.0; length = D))         # spectrum ⊂ [-1, 1]
        f!(Y, X) = mul!(Y, A, X)
        mp = build_mapping_plan(MappingConfig(Emin = -2.0, Emax = 2.0,
                                              optimize_mapping = :universal,
                                              parallel_strategy = BlockWise()),
                                f!, randn(D, 2))
        mr = MappingReport(mp, RowLayout())
        @test mr.center ≈ mp.b
        @test mr.spread ≈ mp.a
        @test mr.Emin ≈ -2.0 && mr.Emax ≈ 2.0
        @test mr.layout == "RowLayout"
        @test mr.dim == D
        @test mr.block_size == 2
        @test mr.optimize_mapping === :universal

        # A vector seed → block size 1; column layout label.
        mp1 = build_mapping_plan(MappingConfig(Emin = -2.0, Emax = 2.0), f!, randn(D))
        mr1 = MappingReport(mp1, ColumnLayout())
        @test mr1.block_size == 1
        @test mr1.layout == "ColumnLayout"
        @test mr1.optimize_mapping === false
    end

    @testset "render_mapping content" begin
        mr = MappingReport(:universal, 0.12, 6.03, -5.91, 6.15, "RowLayout", 4, 4096, "BlockWise")
        s = sprint(io -> render_mapping(io, Formatter(false), mr))
        @test occursin("Mapping", s)
        @test occursin("value-grouped kernel :universal", s)
        # The rescaling is fully determined by the bounds (center = (Emax+Emin)/2,
        # spread = (Emax−Emin)/2), so the section prints the bounds rather than restating both.
        @test occursin("Emin = -5.910", s)
        @test occursin("Emax = 6.150", s)
        @test occursin("opdim 4096", s)
        @test occursin("block size 4", s)
        @test occursin("BlockWise", s)
        @test !occursin("\e[", s)                          # plain: no ANSI escapes
    end
end
