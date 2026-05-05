using LinearAlgebra
using SparseArrays
using Test

using Polfed
using Polfed: MulColsParallel, NoParallel, TwoLevelParallel
using Polfed: optimized_clenshaw_final_sum!, optimized_clenshaw_recurrence_relation!, optimized_mapping!
using Polfed.PolfedCore: get_diags_and_offdiagonals_by_value

function structured_test_matrix()
    H = spzeros(5, 5)
    H[1, 1] = 1.0
    H[2, 2] = 2.0
    H[3, 3] = 3.0
    H[4, 4] = 4.0
    H[5, 5] = 5.0

    H[1, 2] = H[2, 1] = -2.0
    H[2, 4] = H[4, 2] = -2.0
    H[4, 5] = H[5, 4] = -2.0

    H[1, 3] = H[3, 1] = 0.5
    H[2, 5] = H[5, 2] = 0.5

    return H
end

function assert_packed_equal(left, right)
    @test length(left) == length(right)
    for (a, b) in zip(left, right)
        @test a[1] == b[1]
        @test a[2] == b[2]
        @test a[3] == b[3]
    end
end

@testset "optimized mapping internals" begin
    H = structured_test_matrix()
    diagonals, offdiagonals = get_diags_and_offdiagonals_by_value(H)

    @testset "packing sparse matrices" begin
        @test diagonals == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test length(offdiagonals) == 2

        val1, flat1, starts1 = offdiagonals[1]
        val2, flat2, starts2 = offdiagonals[2]

        @test val1 == -2.0
        @test flat1 == [2, 1, 4, 2, 5, 4]
        @test starts1 == [1, 2, 4, 4, 6]

        @test val2 == 0.5
        @test flat2 == [3, 5, 1, 2]
        @test starts2 == [1, 2, 3, 4, 4]
    end

    @testset "packing dense matrices matches sparse matrices" begin
        dense_diagonals, dense_offdiagonals = get_diags_and_offdiagonals_by_value(Matrix(H))

        @test dense_diagonals == diagonals
        assert_packed_equal(dense_offdiagonals, offdiagonals)
    end

    @testset "optimized mapping matches matrix multiplication" begin
        x = Float64[1, 2, 3, 4, 5]
        X = hcat(x, 2 .* x)

        for strategy in (NoParallel(), TwoLevelParallel(1), MulColsParallel(2))
            map! = optimized_mapping!(diagonals, offdiagonals, strategy)

            y = similar(x)
            map!(y, x)
            @test y ≈ H * x

            Y = similar(X)
            map!(Y, X)
            @test Y ≈ H * X
        end
    end

    @testset "optimized Clenshaw steps match direct formulas" begin
        c = 0.5
        x = Float64[1, 2, 3, 4, 5]
        b1 = Float64[5, 4, 3, 2, 1]
        b2 = Float64[4, 3, 2, 1, 0]
        b3 = Float64[1, 1, 1, 1, 1]

        for strategy in (NoParallel(), TwoLevelParallel(1), MulColsParallel(2))
            recurrence! = optimized_clenshaw_recurrence_relation!(diagonals, offdiagonals, strategy)
            out = similar(x)
            recurrence!(out, b2, b3, c, 1, x)
            @test out ≈ c .* x .+ 2 .* (H * b2) .- b3

            final_sum! = optimized_clenshaw_final_sum!(diagonals, offdiagonals, strategy)
            y = similar(x)
            final_sum!(b1, b2, c, y, x)
            @test y ≈ c .* x .+ H * b1 .- b2
        end
    end

    @testset "single offdiagonal bucket mapping" begin
        H_single = spzeros(3, 3)
        H_single[1, 1] = 1.0
        H_single[2, 2] = 2.0
        H_single[3, 3] = 3.0
        H_single[1, 2] = H_single[2, 1] = -0.5
        H_single[2, 3] = H_single[3, 2] = -0.5

        single_diagonals, single_offdiagonals = get_diags_and_offdiagonals_by_value(H_single)
        @test length(single_offdiagonals) == 1

        x = [1.0, 2.0, 3.0]
        for strategy in (NoParallel(), TwoLevelParallel(1), MulColsParallel(2))
            map! = optimized_mapping!(single_diagonals, only(single_offdiagonals), strategy)
            y = similar(x)
            map!(y, x)
            @test y ≈ H_single * x
        end
    end

    @testset "complex CPU optimized mapping" begin
        H_complex = sparse(ComplexF64[
            1.0 + 0.0im  1.0 + 2.0im  0.0 + 0.0im
            1.0 - 2.0im  2.0 + 0.0im -0.5 + 0.0im
            0.0 + 0.0im -0.5 + 0.0im  3.0 + 0.0im
        ])
        complex_diagonals, complex_offdiagonals = get_diags_and_offdiagonals_by_value(H_complex)

        x = ComplexF64[1.0 + 0.5im, -2.0 + 0.25im, 0.75 - 1.0im]
        map! = optimized_mapping!(complex_diagonals, complex_offdiagonals, NoParallel())
        y = similar(x)
        map!(y, x)

        @test y ≈ H_complex * x
    end
end
