# EdgeFilter unit tests. Ground truth is the closed-form polynomial each mode implements —
#   :lowest → −((1−x)/2)^p,  :highest → ((1+x)/2)^p,  :both → x^p (p odd)
# — evaluated through the full Clenshaw (:Taylor/Horner) machinery via `_scalar_filter`, so the
# coefficient table, family wiring, and normalization (exact ±1 peaks) are all checked at once.

using Test
import UniPolfed: EdgeFilter, Defaults, Interval, build_filter, validate_filter,
               exposed_end, polynomial_type, polynomial_order, coefficients, _scalar_filter,
               order_safety_factor, expected_krylov_dim, resolve_eig_target,
               DeltaUpFilter, build_transform
import UniPolfed.Numerics.Lanczos: BothEnds, LowestEigs, HighestEigs

@testset "EdgeFilter" begin
    xs = range(-1.0, 1.0; length = 101)

    @testset "polynomial ≡ closed form (order p = $p)" for p in (1, 2, 3, 7)
        f_lo = EdgeFilter(Interval(-1.0, -0.7), :lowest;  order = p)
        f_hi = EdgeFilter(Interval(0.7, 1.0),   :highest; order = p)
        P_lo, P_hi = _scalar_filter(f_lo), _scalar_filter(f_hi)
        @test maximum(abs(P_lo(x) - (-(((1 - x) / 2)^p))) for x in xs) < 1e-14
        @test maximum(abs(P_hi(x) - (((1 + x) / 2)^p)) for x in xs) < 1e-14
        # exact ±1 peak at the exposed edge, exact 0 at the far edge
        @test P_lo(-1.0) == -1.0 && P_lo(1.0) == 0.0
        @test P_hi(1.0) == 1.0 && P_hi(-1.0) == 0.0

        if isodd(p)
            f_bo = EdgeFilter(Interval(-1.0, 1.0), :both; order = p)
            P_bo = _scalar_filter(f_bo)
            @test maximum(abs(P_bo(x) - x^p) for x in xs) < 1e-14
            @test P_bo(-1.0) == -1.0 && P_bo(1.0) == 1.0
        end
    end

    @testset "coefficients: scaled binomials, zero outside 0..p" begin
        c = coefficients(EdgeFilter(Interval(0.7, 1.0), :highest; order = 3))
        @test [c(n) for n in 0:3] ≈ [1, 3, 3, 1] ./ 8
        @test c(-1) == 0.0 && c(4) == 0.0

        c_lo = coefficients(EdgeFilter(Interval(-1.0, -0.7), :lowest; order = 3))
        @test [c_lo(n) for n in 0:3] ≈ [-1, 3, -3, 1] ./ 8     # (x−1)³/8 = −((1−x)/2)³

        c_bo = coefficients(EdgeFilter(Interval(-1.0, 1.0), :both; order = 5))
        @test [c_bo(n) for n in 0:5] == [0, 0, 0, 0, 0, 1]     # Kronecker δ_{n,p}

        # high order stays stable (a naive C(p,n) overflows Int64 at p ≥ 62)
        P70 = _scalar_filter(EdgeFilter(Interval(0.9, 1.0), :highest; order = 70))
        @test P70(1.0) ≈ 1.0 atol = 1e-12
        @test P70(0.0) ≈ 0.5^70 rtol = 1e-10
    end

    @testset "build_filter from geometry" begin
        f = build_filter(EdgeFilter, Interval(-1.0, -0.7))
        @test f isa EdgeFilter && exposed_end(f) === :lowest && f.target == -1.0
        @test f.order == Defaults.filters.edge.order              # fixed knob, nothing resolved

        f2 = build_filter(EdgeFilter, Interval(0.8, 1.0); order = 5)
        @test exposed_end(f2) === :highest && f2.target == 1.0 && f2.order == 5

        # clamped-at-edge bounds (how the pipeline delivers them) count as touching
        @test exposed_end(build_filter(EdgeFilter, Interval(-1.0 + 1e-12, 0.3))) === :lowest

        @test_throws ArgumentError build_filter(EdgeFilter, Interval(-0.5, 0.5))   # interior
        @test_throws ArgumentError build_filter(EdgeFilter, Interval(-1.0, 1.0))   # full spectrum
    end

    @testset "build_filter :both (two edge windows)" begin
        f = build_filter(EdgeFilter, Interval(-1.0, -0.8), Interval(0.8, 1.0); order = 5)
        @test exposed_end(f) === :both && f.order == 5
        @test f.interval == Interval(-1.0, 1.0)                # hull
        @test isnan(f.target)                                  # no single target

        @test_throws ArgumentError build_filter(EdgeFilter, Interval(-0.9, -0.8), Interval(0.8, 1.0))  # iv1 off edge
        @test_throws ArgumentError build_filter(EdgeFilter, Interval(-1.0, -0.8), Interval(0.8, 0.9))  # iv2 off edge
        @test_throws ArgumentError build_filter(EdgeFilter, Interval(-1.0, 0.2), Interval(0.1, 1.0))   # overlap
        @test_throws ArgumentError build_filter(EdgeFilter, Interval(-1.0, -0.8), Interval(0.8, 1.0); order = 4)  # even
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError EdgeFilter(Interval(-1.0, -0.7), :lowest; order = 0)
        @test_throws ArgumentError EdgeFilter(Interval(-1.0, -0.7), :leftmost)
        @test_throws ArgumentError EdgeFilter(Interval(-1.0, 1.0), :both; order = 2)
        @test EdgeFilter(Interval(-1.0, -0.7), :lowest; order = 2) isa EdgeFilter  # even OK off :both
    end

    @testset "interface tags + validation" begin
        f = build_filter(EdgeFilter, Interval(-1.0, -0.7))
        @test polynomial_type(f) === :Taylor
        @test polynomial_order(f) == f.order
        @test order_safety_factor(f) == 1.0                    # fixed order ⇒ generic fallback
        @test order_safety_factor(DeltaUpFilter(Interval(-0.1, 0.1))) == 0.97  # resolved filters keep theirs
        for ff in (f, build_filter(EdgeFilter, Interval(0.7, 1.0)),
                   build_filter(EdgeFilter, Interval(-1.0, -0.8), Interval(0.8, 1.0)))
            @test validate_filter(ff)                          # monotone ⇒ always clean
        end
    end

    @testset "eig-target and Krylov budget follow the exposed end" begin
        f_lo = build_filter(EdgeFilter, Interval(-1.0, -0.7))
        f_bo = build_filter(EdgeFilter, Interval(-1.0, -0.8), Interval(0.8, 1.0))
        @test resolve_eig_target(exposed_end(f_lo), 4) === LowestEigs(4)
        @test resolve_eig_target(exposed_end(f_bo), 4) === BothEnds(4, 4)
        # behavior, not constants: grows with howmany/blocksize/η; :both budgets more than one end
        @test expected_krylov_dim(f_lo, 20, 1, 1.0) > expected_krylov_dim(f_lo, 5, 1, 1.0)
        @test expected_krylov_dim(f_lo, 5, 4, 1.0) > expected_krylov_dim(f_lo, 5, 1, 1.0)
        @test expected_krylov_dim(f_lo, 5, 1, 2.0) ≈ 2 * expected_krylov_dim(f_lo, 5, 1, 1.0) atol = 1
        @test expected_krylov_dim(f_bo, 5, 1, 1.0) > expected_krylov_dim(f_lo, 5, 1, 1.0)
    end
end
