using Test
using UniPolfed

@testset "build_expansion" begin
    @testset "normalize_clenshaw_recurrence accepts 5-argument recurrence" begin
        called = Ref(false)
        recurrence5!(b1, b2, _b3, c, X) = begin
            called[] = true
            @. b1 = b2 + c * X
            return nothing
        end

        x0 = ones(Float64, 4)
        recurrence! = UniPolfed.normalize_clenshaw_recurrence(
            recurrence5!,
            x0,
            UniPolfed.BlockWise(),
        )

        b1 = zeros(4)
        b2 = fill(2.0, 4)
        b3 = fill(3.0, 4)
        X = fill(5.0, 4)

        @test recurrence!(b1, b2, b3, 0.5, 7, X) === nothing
        @test called[]
        @test b1 ≈ b2 .+ 0.5 .* X
    end

    @testset "normalize_clenshaw_recurrence accepts 6-argument recurrence" begin
        seen_k = Ref(0)
        recurrence6!(b1, b2, _b3, c, k, X) = begin
            seen_k[] = k
            @. b1 = b2 + k * c * X
            return nothing
        end

        x0 = ones(Float64, 4)
        recurrence! = UniPolfed.normalize_clenshaw_recurrence(
            recurrence6!,
            x0,
            UniPolfed.BlockWise(),
        )

        b1 = zeros(4)
        b2 = fill(2.0, 4)
        b3 = fill(3.0, 4)
        X = fill(5.0, 4)

        @test recurrence!(b1, b2, b3, 0.5, 7, X) === nothing
        @test seen_k[] == 7
        @test b1 ≈ b2 .+ 7 .* 0.5 .* X
    end

    @testset "normalize_clenshaw_recurrence rejects invalid recurrence" begin
        x0 = ones(Float64, 4)
        bad_recurrence!(_b1, _b2) = nothing

        @test_throws ArgumentError UniPolfed.normalize_clenshaw_recurrence(
            bad_recurrence!,
            x0,
            UniPolfed.BlockWise(),
        )
    end
end
