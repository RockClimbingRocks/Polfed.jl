# DeltaUpSmoothFilter: construction, the interface methods, and the order/passband/validation logic.
# Same delta-peaked Chebyshev polynomial as DeltaUpFilter, but with a Jackson damping kernel applied
# to the coefficients (a "smooth" delta): the +1 peak stays at the target and everything else is
# suppressed below `cutoff`, but the Gibbs side lobes are damped. Ground truth is the (damped,
# normalized) polynomial itself — peaking at exactly +1 at the target and dropping to `cutoff` at
# the passband edges.

using Test
using UniPolfed

@testset "DeltaUpSmoothFilter - construction" begin

    @testset "Unbuilt filter has negative order" begin
        iv = UniPolfed.Interval(-0.1, 0.1)
        f = UniPolfed.DeltaUpSmoothFilter(iv)
        @test f.order < 0
    end

    @testset "Filter interval is set correctly" begin
        iv = UniPolfed.Interval(-0.1, 0.1)
        f = UniPolfed.DeltaUpSmoothFilter(iv)
        @test f.interval == iv
    end

    @testset "target is the interval midpoint" begin
        iv = UniPolfed.Interval(0.1, 0.3)
        f = UniPolfed.DeltaUpSmoothFilter(iv)
        @test f.target ≈ (iv.left + iv.right) / 2 atol = 1e-12
    end

end

@testset "DeltaUpSmoothFilter - interface methods" begin

    iv = UniPolfed.Interval(-0.1, 0.1)
    f = UniPolfed.DeltaUpSmoothFilter(iv; order = 25, cutoff = 0.17, order_safety_factor = 0.9)

    @testset "exposed_end is the highest end" begin
        @test UniPolfed.exposed_end(f) == :highest
    end

    @testset "polynomial family is Chebyshev" begin
        @test UniPolfed.polynomial_type(f) == :Chebyshev
    end

    @testset "polynomial_order reports the stored order" begin
        @test UniPolfed.polynomial_order(f) == 25
    end

    @testset "order_safety_factor accessor reports the stored factor" begin
        @test UniPolfed.order_safety_factor(f) == 0.9
    end

    @testset "expected_krylov_dim matches the empirical delta fit" begin
        @test UniPolfed.expected_krylov_dim(f, 10, 1, 1.0) == ceil(Int, 20.427 * 1 + 1.696 * 10)
        @test UniPolfed.expected_krylov_dim(f, 40, 4, 1.3) == ceil(Int, (20.427 * 4 + 1.696 * 40) * 1.3)
        # larger safety factor never shrinks the estimate
        @test UniPolfed.expected_krylov_dim(f, 10, 1, 1.5) >= UniPolfed.expected_krylov_dim(f, 10, 1, 1.0)
    end

end
