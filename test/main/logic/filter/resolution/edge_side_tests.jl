# edge_side / _check_interval: the window-geometry decision resolve_filter branches on. A window
# TOUCHING ±1 is edge territory; a near-edge window like [-0.99, -0.9] is deliberately :interior
# (the delta cutoff ladder handles it); the full spectrum is rejected.

import Polfed: edge_side, _check_interval, Interval

@testset "edge_side" begin
    @testset "classification" begin
        @test edge_side(Interval(-0.1, 0.1)) === :interior
        @test edge_side(Interval(-1.0, -0.9)) === :left
        @test edge_side(Interval(0.9, 1.0)) === :right
        @test edge_side(Interval(-1.0, 0.3)) === :left            # wide, still on the edge
        # near-edge but NOT touching stays interior — the delta-ladder case
        @test edge_side(Interval(-0.99, -0.9)) === :interior
        @test edge_side(Interval(0.9, 0.99)) === :interior
        # a clamped bound lands within rounding of ±1 and counts as touching
        @test edge_side(Interval(-1.0 + 1e-12, -0.9)) === :left
    end

    @testset "atol widens/narrows the edge band" begin
        @test edge_side(Interval(-0.99, -0.9); atol = 0.05) === :left
        @test edge_side(Interval(-1.0 + 1e-12, -0.9); atol = 0.0) === :interior
        @test_throws ArgumentError edge_side(Interval(-0.1, 0.1); atol = -1.0)
    end

    @testset "full spectrum and malformed intervals are rejected" begin
        @test_throws ArgumentError edge_side(Interval(-1.0, 1.0))          # full spectrum
        @test_throws ArgumentError _check_interval(Interval(-1.5, 0.0))    # below -1
        @test_throws ArgumentError _check_interval(Interval(0.0, 1.5))     # above +1
        @test_throws ArgumentError _check_interval(Interval(0.5, 0.5))     # empty
        @test_throws ArgumentError _check_interval(Interval(0.5, 0.2))     # inverted
        @test_throws ArgumentError _check_interval(Interval(-Inf, 0.0))    # non-finite
    end
end
