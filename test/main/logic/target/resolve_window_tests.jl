
using Test
using Polfed
using Polfed: Interval, TargetRescaledWindow, TargetAbsoluteWindow, resolve_window,
              howmany_in_interval, DoSResult

@testset "resolve_window" begin

    @testset "rescaled window → interval verbatim (a, b ignored)" begin
        iv = resolve_window(TargetRescaledWindow(0.2, 0.8), 1.0, 0.0)
        @test iv isa Interval
        @test iv.left ≈ 0.2 && iv.right ≈ 0.8
        iv2 = resolve_window(TargetRescaledWindow(-1.0, 1.0), 1.0, 0.0)   # boundary allowed
        @test iv2.left ≈ -1.0 && iv2.right ≈ 1.0
    end

    @testset "rescaled window outside [-1,1] throws" begin
        @test_throws ArgumentError resolve_window(TargetRescaledWindow(0.5, 1.5), 1.0, 0.0)
        @test_throws ArgumentError resolve_window(TargetRescaledWindow(-1.5, 0.5), 1.0, 0.0)
    end

    @testset "unrescaled window rescaled via (E - b)/a" begin
        # Emin=-2, Emax=2 ⇒ a=2, b=0.  E∈[-1,1] → rescaled [-0.5, 0.5].
        iv = resolve_window(TargetAbsoluteWindow(-1.0, 1.0), 2.0, 0.0)
        @test iv.left ≈ -0.5 && iv.right ≈ 0.5
        # asymmetric: Emin=0, Emax=10 ⇒ a=5, b=5.  E∈[2,7] → rescaled [-0.6, 0.4].
        iv2 = resolve_window(TargetAbsoluteWindow(2.0, 7.0), 5.0, 5.0)
        @test iv2.left ≈ -0.6 && iv2.right ≈ 0.4
    end

    @testset "unrescaled window overhanging the spectrum clamps + warns" begin
        iv = @test_logs (:warn, r"exceeds the estimated spectrum") resolve_window(
                 TargetAbsoluteWindow(-3.0, 3.0), 2.0, 0.0)          # [-1.5,1.5] → clamp [-1,1]
        @test iv.left ≈ -1.0 && iv.right ≈ 1.0
        iv2 = @test_logs (:warn, r"exceeds") resolve_window(          # one-sided overhang
                  TargetAbsoluteWindow(-3.0, 1.0), 2.0, 0.0)
        @test iv2.left ≈ -1.0 && iv2.right ≈ 0.5
    end

    @testset "unrescaled window entirely off the spectrum → empty after clamp throws" begin
        @test_logs begin
            @test_throws ArgumentError resolve_window(TargetAbsoluteWindow(3.0, 4.0), 2.0, 0.0)
        end
    end
end

@testset "howmany_in_interval" begin
    # Synthetic DoS: only the k=0 Chebyshev coefficient (moments/ρ are unused here).
    dos   = DoSResult([1.0], [1.0], x -> 1.0)
    icw(l, r) = Polfed.integrate_chebyshev_weighted([1.0], l, r) / π

    @testset "count + buffer wiring (default buffer = 2·blocksize)" begin
        for (l, r) in [(-1.0, 1.0), (0.0, 1.0), (-0.3, 0.3)]
            expected = max(1, ceil(Int, icw(l, r)))
            @test howmany_in_interval(dos, Interval(l, r)) == expected
        end
    end

    @testset "monotonic in window width; ≥ 1 even razor-thin" begin
        @test howmany_in_interval(dos, Interval(-0.2,0.2)) <=
              howmany_in_interval(dos, Interval(-0.8,0.8))
        @test howmany_in_interval(dos, Interval(0.10, 0.1001)) >= 1
    end
end
