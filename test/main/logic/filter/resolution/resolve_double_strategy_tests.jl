# Tests for the joint-vs-separate decision behind `double_polfed`. Pure geometry + filter
# resolution — no operator, no Lanczos — so the cost model is asserted directly on the numbers it
# is made of, not inferred from a run's timing.

import Polfed: resolve_double_strategy, DoubleStrategy, double_order, Interval,
               DoubleDeltaFilter, EdgeFilter, polynomial_order, Defaults

@testset "resolve_double_strategy" begin
    @testset "the ratio rule and its threshold agree with the Krylov overhead" begin
        # The threshold is not free: the joint run costs `K_d·m_d` against `K₁m₁ + K₂m₂`, so with
        # m_d = krylov_overhead·m the joint run wins exactly when overhead < 1 + r. Locking the
        # relation here is what stops the two constants drifting apart.
        @test Defaults.double_polfed.double_ratio ≈
              Defaults.double_polfed.krylov_overhead - 1.0
    end

    @testset "two comparable windows → one joint run" begin
        # Symmetric placement, equal widths ⇒ K₁ ≈ K₂ ⇒ ratio ≈ 1 ⇒ well above the threshold.
        low, high = Interval(-0.5, -0.4), Interval(0.4, 0.5)
        s = resolve_double_strategy(low, high)
        @test s.mode === :double
        @test s.reason === :ratio
        @test s.double_filter isa DoubleDeltaFilter
        @test s.ratio >= Defaults.double_polfed.double_ratio
        @test double_order(s) == polynomial_order(s.double_filter)
        @test s.order1 > 0 && s.order2 > 0
    end

    @testset "very unequal orders → two separate runs" begin
        # A wide window needs a low order, a narrow one a high order (K ≈ 2.655·√(1−λ²)/δ). Made
        # lopsided enough that the ratio must fall under the threshold.
        wide, narrow = Interval(-0.60, -0.10), Interval(0.400, 0.404)
        s = resolve_double_strategy(wide, narrow)
        @test s.ratio < Defaults.double_polfed.double_ratio
        @test s.mode === :separate
        @test s.reason === :ratio
        @test s.double_filter === nothing
        @test double_order(s) == 0
        @test s.order1 != s.order2          # the asymmetry that drove the decision
    end

    @testset "forcing overrides the ratio, in both directions" begin
        low, high = Interval(-0.5, -0.4), Interval(0.4, 0.5)      # ratio would say :double
        forced = resolve_double_strategy(low, high, :separate)
        @test forced.mode === :separate && forced.reason === :forced
        @test forced.ratio >= Defaults.double_polfed.double_ratio  # decided against the ratio

        wide, narrow = Interval(-0.60, -0.10), Interval(0.400, 0.404)  # ratio would say :separate
        joint = resolve_double_strategy(wide, narrow, :double)
        @test joint.ratio < Defaults.double_polfed.double_ratio
        @test joint.mode === :double && joint.reason === :forced
    end

    @testset "an edge-touching window cannot be served jointly" begin
        # `±1` is EdgeFilter territory (a delta peak cannot sit there), so the pair falls back to
        # two runs however good the ratio is — the decision is advisory, the geometry is not.
        edge, interior = Interval(-1.0, -0.9), Interval(0.4, 0.5)
        s = resolve_double_strategy(edge, interior, :double)   # forced, and still overruled
        @test s.mode === :separate
        @test s.reason === :edge
        @test s.filter1 isa EdgeFilter
    end

    @testset "the separate path returns usable single-window filters" begin
        wide, narrow = Interval(-0.60, -0.10), Interval(0.400, 0.404)
        s = resolve_double_strategy(wide, narrow)
        @test polynomial_order(s.filter1) == s.order1
        @test polynomial_order(s.filter2) == s.order2
    end

    @testset "malformed input" begin
        @test_throws ArgumentError resolve_double_strategy(
            Interval(-0.5, -0.4), Interval(0.4, 0.5), :sideways)      # unknown strategy
        @test_throws ArgumentError resolve_double_strategy(
            Interval(0.4, 0.5), Interval(-0.5, -0.4))                 # unordered
        @test_throws ArgumentError resolve_double_strategy(
            Interval(-0.5, 0.1), Interval(0.0, 0.5))                  # overlapping
    end
end
