# DeltaUpSmoothFilter coefficients. Ground truth is the Jackson-damped, normalized Chebyshev
# expansion of the delta peak at the target. Two families of properties:
#   • structure at target = 0 (shared with the sharp delta): odd coefficients vanish and the even
#     coefficients alternate sign — the Jackson kernel g_n > 0 leaves both intact;
#   • what damping CHANGES: the even-coefficient magnitudes are no longer equal (as they are for the
#     undamped delta) but strictly DECREASE with n, because g_n decreases monotonically from g_0 = 1
#     to g_order ≈ 0.
# In every case the peak is normalized to exactly +1 at the target.

using Test
using UniPolfed

@testset "DeltaUpSmoothFilter - coefficients" begin

    @testset "target = 0.0 -> odd coefficients = 0.0" begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaUpSmoothFilter(iv; order = 25, cutoff = 0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)

        odd_coefs_zero = true
        for n in 0:filter.order
            if isodd(n)
                odd_coefs_zero_n = isapprox(coefs(n), 0.0, atol = 1e-14) ? true : false
                odd_coefs_zero = odd_coefs_zero && odd_coefs_zero_n
            end
        end

        @test odd_coefs_zero
    end

    @testset "target = 0.0 -> even coeffs (n>0) alternate sign" begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaUpSmoothFilter(iv; order = 25, cutoff = 0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)

        even_coefs_change_sign1 = true
        even_coefs_change_sign2 = true

        for n in 0:filter.order
            if iseven(n) && n > 0
                even_coefs_change_sign_n1 = sign(coefs(n - 2)) != sign(coefs(n)) ? true : false
                even_coefs_change_sign1 = even_coefs_change_sign1 && even_coefs_change_sign_n1
                if n >= 4
                    even_coefs_change_sign_n2 = sign(coefs(n - 4)) == sign(coefs(n)) ? true : false
                    even_coefs_change_sign2 = even_coefs_change_sign2 && even_coefs_change_sign_n2
                end
            end
        end

        @test even_coefs_change_sign1
        @test even_coefs_change_sign2
    end

    @testset "target = 0.0 -> even |coeffs| strictly decrease with n (Jackson damping)" begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaUpSmoothFilter(iv; order = 25, cutoff = 0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)

        # g_n decreases monotonically, so for even n ≥ 2 each magnitude is strictly below the previous
        # even one. (For the undamped delta these are all equal — this is the damping's signature.)
        even_coefs_decreasing = true
        for n in 4:2:filter.order
            even_coefs_decreasing = even_coefs_decreasing && (abs(coefs(n)) < abs(coefs(n - 2)))
        end

        @test even_coefs_decreasing
    end

    @testset "target = 0.0 -> damping shrinks c2 below the undamped 2·|c0|" begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaUpSmoothFilter(iv; order = 25, cutoff = 0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)

        # The undamped delta has |c2| = 2·|c0| exactly; the Jackson kernel (g_2 < 1) pulls it below.
        @test abs(coefs(2)) < 2 * abs(coefs(0))
    end

    @testset "coefficients normalized to P(target) = 1.0" begin
        for target in [0.0, 1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.9]
            δ = 1e-2
            iv = UniPolfed.Interval(target - δ, target + δ)
            filter = UniPolfed.DeltaUpSmoothFilter(iv; order = 25, cutoff = 0.17, order_safety_factor = 1)
            P = UniPolfed._scalar_filter(filter)

            @test P(target) ≈ 1.0 atol = 1e-10
        end
    end
end
