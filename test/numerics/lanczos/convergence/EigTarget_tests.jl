# Types-only layer of the convergence subsystem: the `EndSpec`/`EigTarget` structs, the
# aliases, `ends` (which is what lets Lowest/Highest/BothEnds share one code path
# downstream), and `requested_count`. The rules themselves live in `end_spec_tests.jl`,
# the walker in `converged_run_tests.jl`.

using Test
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

@testset "EigTarget" begin

    @testset "aliases" begin
        @test LowestEigs(3)  === Lowest(NumEigs(3))
        @test HighestEigs(3) === Highest(NumEigs(3))
        @test BothEnds(2, 3) === BothEnds(NumEigs(2), NumEigs(3))
    end

    @testset "NumEigs requires a positive count" begin
        @test NumEigs(Int32(2)) == NumEigs(2)
        @test_throws ArgumentError NumEigs(0)
        @test_throws ArgumentError NumEigs(-1)
        @test_throws ArgumentError LowestEigs(0)
        @test_throws ArgumentError HighestEigs(-1)
        @test_throws ArgumentError BothEnds(1, 0)
    end

    @testset "Window struct: promote + low<high guard" begin
        w = Window(0.5, 0.6)
        @test (w.low, w.high) == (0.5, 0.6)
        @test Window(0, 0.6) isa Window{Float64}          # Int/Float promote
        @test_throws ArgumentError Window(0.6, 0.5)
        @test_throws ArgumentError Window(0.5, 0.5)
        @test Window(-Inf, 0.6) isa Window{Float64}     # edge windows are legal

        certified = Window(0.2, 0.8, 0.5)
        @test certified.edge_value == 0.5
        @test !certified.return_brackets
        @test Window(0.2, 0.8).return_brackets
        @test_throws ArgumentError Window(0.2, 0.8, Inf)
    end

    @testset "ends — the single seam Lowest/Highest/BothEnds share" begin
        # One (spec, islow) entry per targeted extreme. `islow` picks the walk direction.
        @test LZ.ends(Lowest(NumEigs(3)))  == ((NumEigs(3), true),)
        @test LZ.ends(Highest(NumEigs(3))) == ((NumEigs(3), false),)
        @test LZ.ends(BothEnds(2, 3))      == ((NumEigs(2), true), (NumEigs(3), false))

        # Spec type is irrelevant to `ends` — that is the point.
        @test LZ.ends(Highest(ToValue(0.5)))  == ((ToValue(0.5), false),)
        @test length(LZ.ends(BothEnds(Window(-0.6, -0.5), Window(0.5, 0.6)))) == 2
    end

    @testset "requested_count — dynamic targets report -1" begin
        @test LZ.requested_count(LowestEigs(4))  == 4
        @test LZ.requested_count(HighestEigs(4)) == 4
        @test LZ.requested_count(BothEnds(2, 3)) == 5
        # Value-/window-based counts are only known at run time.
        @test LZ.requested_count(Highest(ToValue(0.0)))            == -1
        @test LZ.requested_count(Highest(Window(0.5, 0.6)))    == -1
        @test LZ.requested_count(BothEnds(NumEigs(2), ToValue(0.0))) == -1
    end

    @testset "type stability" begin
        @test @inferred(LZ.ends(HighestEigs(3))) isa Tuple
        @test @inferred(LZ.requested_count(HighestEigs(3))) isa Int
    end
end
