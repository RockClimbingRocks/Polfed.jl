# bisect_order: the shared bracket-double + bisect order-resolution mechanic. Driven by a
# caller-supplied `suppressed(order)::Bool` predicate, so we test it against synthetic monotone
# predicates (independent of any filter).
using Test
using Polfed

@testset "bisect_order" begin
    @testset "order 1 already suppresses" begin
        @test Polfed.bisect_order(2.0, _ -> true) == 1
    end

    @testset "returns the first suppressing order exactly" begin
        for thresh in (7, 33, 128)
            suppressed = k -> k >= thresh
            K = Polfed.bisect_order(2.0, suppressed)          # analytical bracket deliberately too small
            @test suppressed(K)                        # the returned order does suppress
            @test K == thresh                          # integer bisection returns the first true order
        end
    end

    @testset "threshold immediately after order 1" begin
        @test Polfed.bisect_order(2.0, k -> k >= 2) == 2
    end

    @testset "analytical estimate may start above the threshold" begin
        @test Polfed.bisect_order(100.0, k -> k >= 7) == 7
    end

    @testset "brackets far beyond the analytical estimate (doubling)" begin
        K = Polfed.bisect_order(2.0, k -> k >= 5000)
        @test K == 5000
    end

    @testset "errors when nothing suppresses" begin
        @test_throws ErrorException Polfed.bisect_order(2.0, _ -> false)
    end
end
