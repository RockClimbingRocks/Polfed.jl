
using Test
using UniPolfed

@testset "DeltaDownFilter - coefficients" begin
    
    @testset "target = 0.0 -> odd coefficients = 0.0 " begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaDownFilter(iv; order = 25, cutoff = -0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)


        odd_coefs_zero = true
        for n in 0:filter.order
            if isodd(n)
                odd_coefs_zero_n = isapprox(coefs(n), 0.0, atol=1e-14) ? true : false
                odd_coefs_zero = odd_coefs_zero && odd_coefs_zero_n
            end
        end

        @test odd_coefs_zero 
    end

    @testset "target = 0.0 -> even coeffs for n>0 are of the same magnitude " begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaDownFilter(iv; order = 25, cutoff = -0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)



        even_coefs_are_the_same = true
        for n in 0:filter.order
            if iseven(n) && n > 0
                even_coefs_are_the_same_n = isapprox(abs(coefs(n)), abs(coefs(2)), atol=1e-14) ? true : false
                even_coefs_are_the_same = even_coefs_are_the_same && even_coefs_are_the_same_n        
            end
        end

        @test even_coefs_are_the_same
    end

    @testset "target = 0.0 -> even coeffs for n>0 are of the same magnitude " begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaDownFilter(iv; order = 25, cutoff = -0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)



        even_coefs_are_the_same = true
        for n in 0:filter.order
            if iseven(n) && n > 0
                even_coefs_are_the_same_n = isapprox(abs(coefs(n)), abs(coefs(2)), atol=1e-14) ? true : false
                even_coefs_are_the_same = even_coefs_are_the_same && even_coefs_are_the_same_n        
            end
        end

        @test even_coefs_are_the_same
    end

    @testset "target = 0.0 -> even coeffs for n=0 half of the magnitude" begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaDownFilter(iv; order = 25, cutoff = -0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)



        even_coefs_half_magnitude = true

        for n in 0:filter.order
            if iseven(n) && n > 0
                even_coefs_half_magnitude_n = isapprox(2*abs(coefs(0)), abs(coefs(n)), atol=1e-14) ? true : false
                even_coefs_half_magnitude = even_coefs_half_magnitude && even_coefs_half_magnitude_n        
            end
        end

        @test even_coefs_half_magnitude
    end

    @testset "target = 0.0 -> even coeffs change of sign" begin
        δ = 1e-2
        iv = UniPolfed.Interval(-δ, δ)
        filter = UniPolfed.DeltaDownFilter(iv; order = 25, cutoff = -0.17, order_safety_factor = 1)
        coefs = UniPolfed.coefficients(filter)


        even_coefs_change_sign1 = true
        even_coefs_change_sign2 = true

        for n in 0:filter.order
            if iseven(n) && n > 0
                even_coefs_change_sign_n1 = sign(coefs(n-2)) != sign(coefs(n)) ? true : false
                even_coefs_change_sign1 = even_coefs_change_sign1 && even_coefs_change_sign_n1 
                if n>=4
                    even_coefs_change_sign_n2 = sign(coefs(n-4)) == sign(coefs(n)) ? true : false
                    even_coefs_change_sign2 = even_coefs_change_sign2 && even_coefs_change_sign_n2        
                end    
            end
        end

        @test even_coefs_change_sign1
        @test even_coefs_change_sign2
    end

    @testset "coefficents normalized to P(target) = -1.0" begin

        for target in [0., 1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.9]
            δ = 1e-2
            iv = UniPolfed.Interval(target-δ, target+δ)
            filter = UniPolfed.DeltaDownFilter(iv; order = 25, cutoff = -0.17, order_safety_factor = 1)
            P = UniPolfed._scalar_filter(filter)

            @test P(target) ≈ -1.0 atol = 1e-10
        end
    end
end
