# DeltaUpFilter: construction, the interface methods, and the order/passband/validation logic.
# Ground truth is the delta-peaked Chebyshev polynomial itself — normalized to peak at exactly +1
# at the target, dropping to `cutoff` at the passband edges.

using Test
using UniPolfed

@testset "DeltaDownFilter - construction" begin

    @testset "Unbuilt filter has negative order" begin
        iv = UniPolfed.Interval(-0.1, 0.1)
        f = UniPolfed.DeltaDownFilter(iv)
        @test f.order < 0
    end

    @testset "Filter interval is set correctly" begin
        iv = UniPolfed.Interval(-0.1, 0.1)
        f = UniPolfed.DeltaDownFilter(iv) 
        @test f.interval == iv
    end

end
