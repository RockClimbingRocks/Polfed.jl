# DeltaUpSmoothFilter: the Jackson-damped delta filter. Asserted against mathematical ground
# truth (kernel identities, peak normalization, lobe suppression) rather than snapshots, and
# against the sharp filter it is derived from.

using Polfed: Interval, DeltaUpFilter, DeltaUpSmoothFilter, Defaults,
                 build_filter, validate_filter, _scalar_filter, _jackson_kernel,
                 coefficients, passband, cutoff, exposed_end, polynomial_order,
                 polynomial_type, order_safety_factor, expected_krylov_dim,
                 resolve_filter

@testset "DeltaUpSmoothFilter" begin

    @testset "Jackson kernel identities" begin
        for order in (4, 17, 60)
            # g₀ = 1 exactly: the kernel must not rescale the constant term
            @test _jackson_kernel(0, order) ≈ 1.0 atol = 1e-12
            # monotonically decreasing, and non-negative across the series
            gs = [_jackson_kernel(n, order) for n in 0:order]
            @test all(gs .>= -1e-12)
            @test issorted(gs; rev = true)
            # decays toward 0 at the truncation point (slowly for very short series:
            # g_end ≈ 0.083 at order 4, ≈ 8e-5 at order 60)
            @test gs[end] < 0.1
            @test gs[end] < gs[1]
        end
    end

    @testset "peaks at exactly +1 on the target" begin
        for (lam, hw) in ((0.0, 0.10), (0.25, 0.06), (-0.4, 0.08))
            f = build_filter(DeltaUpSmoothFilter, Interval(lam - hw, lam + hw))
            P = _scalar_filter(f)
            @test P(f.target) ≈ 1.0 atol = 1e-10
        end
    end

    @testset "type-tag interface matches the sharp filter" begin
        f = build_filter(DeltaUpSmoothFilter, Interval(-0.1, 0.1))
        @test exposed_end(f) === :highest
        @test polynomial_type(f) === :Chebyshev
        @test polynomial_order(f) == f.order
        @test cutoff(f) == f.cutoff
        @test passband(f) isa Interval
        @test order_safety_factor(f) == Defaults.filters.delta_up_smooth.order_safety_factor
        # same empirical Krylov fit as the sharp delta filters
        sharp = build_filter(DeltaUpFilter, Interval(-0.1, 0.1))
        @test expected_krylov_dim(f, 200, 1, 1.25) == expected_krylov_dim(sharp, 200, 1, 1.25)
    end

    @testset "damping suppresses the side lobes it exists to remove" begin
        for (lam, hw) in ((0.0, 0.10), (0.2, 0.05), (-0.35, 0.08))
            iv = Interval(lam - hw, lam + hw)
            sharp  = build_filter(DeltaUpFilter, iv)
            smooth = build_filter(DeltaUpSmoothFilter, iv)
            Ps, Pm = _scalar_filter(sharp), _scalar_filter(smooth)

            farlobe(P, f) = begin
                w = f.passband.right - f.passband.left
                m = 0.0
                for x in range(-1, 1; length = 4000)
                    abs(x - f.target) < 3w && continue
                    m = max(m, abs(P(x)))
                end
                m
            end

            # at least an order of magnitude quieter far from the peak
            @test farlobe(Pm, smooth) < farlobe(Ps, sharp) / 10
        end
    end

    @testset "the damping is paid for in order" begin
        for (lam, hw) in ((0.0, 0.10), (0.2, 0.05), (-0.35, 0.08))
            iv = Interval(lam - hw, lam + hw)
            sharp  = build_filter(DeltaUpFilter, iv)
            smooth = build_filter(DeltaUpSmoothFilter, iv)
            # a wider damped peak needs a higher order for the same window — roughly 2x
            @test smooth.order > sharp.order
            @test 1.5 < smooth.order / sharp.order < 3.0
        end
    end

    @testset "resolved filters are valid and cover their request" begin
        for (lam, hw) in ((0.0, 0.10), (0.25, 0.06), (-0.4, 0.08), (0.5, 0.04))
            iv = Interval(lam - hw, lam + hw)
            f = build_filter(DeltaUpSmoothFilter, iv)
            @test validate_filter(f)
            @test passband(f).left  <= iv.left  + 1e-8
            @test passband(f).right >= iv.right - 1e-8
        end
    end

    @testset "sits last in the interior chain, so it stays a fallback" begin
        @test DeltaUpSmoothFilter in Defaults.filters.INTERIOR_FILTER_CHAIN
        @test Defaults.filters.INTERIOR_FILTER_CHAIN[end] === DeltaUpSmoothFilter
        # an ordinary interior window is still served by the cheap sharp filter
        @test resolve_filter(Interval(-0.05, 0.05)) isa DeltaUpFilter
    end
end
