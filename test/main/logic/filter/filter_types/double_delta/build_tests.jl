# DoubleDeltaFilter: construction, the interface methods, and the order/passband/validation logic.
# Ground truth is the combined delta-peaked Chebyshev polynomial itself — normalized to peak at
# exactly +1 at target1 and -1 at target2, dropping to the cutoffs at the passband edges.

using Test
using Polfed

@testset "DoubleDeltaFilter construction" begin
    @testset "filter contract for |cutoff|=$(cutoff)" for cutoff in [0.25,0.3,0.35]
        cases = [
            (Polfed.Interval(-0.3, -0.2), Polfed.Interval(+0.2, +0.3)),   # symmetric, moderate
            (Polfed.Interval(-0.5, -0.4), Polfed.Interval(+0.1, +0.3)),   # asymmetric widths, off-center
            (Polfed.Interval(-0.2, -0.1), Polfed.Interval(+0.15, +0.45)), # right window wide
            (Polfed.Interval(-0.7, -0.55), Polfed.Interval(+0.05, +0.15)),# left window near the edge
            (Polfed.Interval(-0.15, -0.05), Polfed.Interval(+0.05, +0.15)),# both windows close to 0
            (Polfed.Interval(-0.45, -0.35), Polfed.Interval(-0.1, +0.2)), # both windows on the left half
            (Polfed.Interval(-0.52, -0.5), Polfed.Interval(0.5, 0.52)),
            (Polfed.Interval(-0.8, -0.799), Polfed.Interval(0.1, +0.101)),
            (Polfed.Interval(-0.001, -0.0005), Polfed.Interval(0.0001, 0.0005)),
            (Polfed.Interval(-0.6, -0.3), Polfed.Interval(+0.3, +0.6)),   # wide symmetric
            (Polfed.Interval(-0.04, -0.02), Polfed.Interval(+0.5, +0.7)), # tiny left, wide right, very asymmetric
            (Polfed.Interval(-0.9, -0.88), Polfed.Interval(+0.88, +0.9)), # both windows near the spectrum edges
        ]

        for case in cases
            iv1, iv2 = case
            f = Polfed.build_filter(
                Polfed.DoubleDeltaFilter, iv1, iv2;
                cutoff_upper = +cutoff,
                cutoff_lower = -cutoff,
                order_safety_factor = 0.97
            )

            P = Polfed._scalar_filter(f)

            # Identities and inequalities that must hold for a valid filter
            @test f.interval1.left          <  f.target1             < f.interval1.right
            @test f.interval2.left          <  f.target2             < f.interval2.right
            @test f.interval1.right         <  f.interval2.left

            @test f.passband1.left          <  f.target1             < f.passband1.right
            @test f.passband2.left          <  f.target2             < f.passband2.right
            @test f.passband1.right         <  f.passband2.left

            @test f.passband1.left          <= f.interval1.left
            @test f.passband1.right         >= f.interval1.right
            @test f.passband2.left          <= f.interval2.left
            @test f.passband2.right         >= f.interval2.right


            @test P(f.interval1.left)    >= f.cutoff_upper
            @test P(f.interval1.right)   >= f.cutoff_upper
            @test P(f.interval2.left)    <= f.cutoff_lower
            @test P(f.interval2.right)   <= f.cutoff_lower

            @test P(f.passband1.left)    ≈  f.cutoff_upper atol = 1e-8
            @test P(f.passband1.right)   ≈  f.cutoff_upper atol = 1e-8

            @test P(f.passband2.left)    ≈  f.cutoff_lower atol = 1e-8
            @test P(f.passband2.right)   ≈  f.cutoff_lower atol = 1e-8

            @test P(f.target1)           ≈  1.0      atol = 1e-8
            @test P(f.target2)           ≈  -1.0     atol = 1e-8

            # Within each passband P stays past its cutoff (amplified) — checked via validate.jl.
            npts_pb = Polfed.Defaults.filters.double_delta.npts_within_passband(f.order1, f.order2)
            @test Polfed.validate_passbands(P, f, Polfed.Defaults.filters.double_delta.Δε, 0.0, npts_pb)
        end


    end


    # Invalid intervals are rejected at construction (build_filter), not deferred to validate_filter:
    # build_filter must never emit a filter from a malformed/mis-ordered window pair.

    @testset "Check validity of first interval" begin
        for cutoff in [0.25, 0.3, 0.35]
            cases = [
                (Polfed.Interval(-0.3, -0.3), Polfed.Interval(-0.2, +0.3)),  # iv1 empty (left == right)
                (Polfed.Interval(-0.3, -0.5), Polfed.Interval(-0.25,-0.15)), # iv1 reversed (left > right)
            ]

            for case in cases
                iv1, iv2 = case
                @test_throws ArgumentError Polfed.build_filter(
                    Polfed.DoubleDeltaFilter, iv1, iv2;
                    cutoff_upper = +cutoff,
                    cutoff_lower = -cutoff,
                    order_safety_factor = 1
                )
            end
        end
    end

    @testset "Check validity of second interval" begin
        for cutoff in [0.25, 0.3, 0.35]
            cases = [
                (Polfed.Interval(-0.4, -0.3), Polfed.Interval(+0.2, +0.1)),  # iv2 reversed
                (Polfed.Interval(-0.8, -0.5), Polfed.Interval(+0.25,-0.05)), # iv2 reversed
            ]

            for case in cases
                iv1, iv2 = case
                @test_throws ArgumentError Polfed.build_filter(
                    Polfed.DoubleDeltaFilter, iv1, iv2;
                    cutoff_upper = +cutoff,
                    cutoff_lower = -cutoff,
                    order_safety_factor = 1
                )
            end
        end
    end

    @testset "Check validity of the intervals - iv1 on the left of iv2" begin
        for cutoff in [0.25, 0.3, 0.35]
            cases = [
                (Polfed.Interval(-0.3, -0.2), Polfed.Interval(-0.2, +0.3)),  # touching (iv1.right == iv2.left)
                (Polfed.Interval(-0.3, -0.2), Polfed.Interval(-0.25,-0.15)), # overlapping
                (Polfed.Interval(+0.2, +0.3), Polfed.Interval(-0.2, +0.3)),  # iv1 to the right of iv2
                (Polfed.Interval(+0.2, +0.3), Polfed.Interval(-0.2, +0.)),   # iv1 to the right of iv2
            ]

            for case in cases
                iv1, iv2 = case
                @test_throws ArgumentError Polfed.build_filter(
                    Polfed.DoubleDeltaFilter, iv1, iv2;
                    cutoff_upper = +cutoff,
                    cutoff_lower = -cutoff,
                    order_safety_factor = 1
                )
            end
        end
    end
end
