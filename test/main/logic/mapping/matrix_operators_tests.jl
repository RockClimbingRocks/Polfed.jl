# The generic operators of the matrix entrypoints (`optimize_mapping = false`), in both layouts:
# `op(Y, X)` plus the accumulating forms AxpbyClenshaw runs on, and the map-back, checked against
# dense arithmetic. Every operator applies H to each Krylov VECTOR — a column, or a row of a
# `bs × D` block (`X·Hᵀ`, not `X·H`: the two differ by a conjugation for a complex Hermitian H).

using Test
using Polfed
using LinearAlgebra, SparseArrays, Random

@testset "generic matrix operators, $T" for T in (Float64, ComplexF64)
    Random.seed!(5)
    D, bs, a, b = 40, 3, 2.5, 0.4
    H = sprandn(T, D, D, 0.2); H = H + H'
    H̃ = (Matrix(H) - b * I) / a

    @testset "row: op(Y, X) applies H̃ to every row" begin
        op = Polfed._generic_rescaled(RowLayout(), H, a, b)
        X = randn(T, bs, D)
        Y = fill(T(NaN), bs, D)
        op(Y, X)
        @test Y ≈ permutedims(H̃ * permutedims(X))
    end

    @testset "row: _axpby! accumulates; β = 0 does not read Y" begin
        op = Polfed._generic_rescaled(RowLayout(), H, a, b)
        X, Y0 = randn(T, bs, D), randn(T, bs, D)
        Y = copy(Y0)
        Polfed._axpby!(op, Y, X, 2.0, -1.0)
        @test Y ≈ 2 .* permutedims(H̃ * permutedims(X)) .- Y0
        Y = fill(T(NaN), bs, D)
        Polfed._axpby!(op, Y, X, 1.5, 0)
        @test Y ≈ 1.5 .* permutedims(H̃ * permutedims(X))
        Y = copy(Y0)
        Polfed._add!(op, Y, 0.3, X)
        @test Y ≈ Y0 .+ 0.3 .* X
    end

    @testset "column: op(Y, X) = H̃·X for a block and a vector" begin
        op = Polfed._generic_rescaled(ColumnLayout(), H, a, b)
        X = randn(T, D, bs)
        Y = fill(T(NaN), D, bs)
        op(Y, X)
        @test Y ≈ H̃ * X
        x, y0 = randn(T, D), randn(T, D)
        y = copy(y0)
        Polfed._axpby!(op, y, x, 2.0, -1.0)
        @test y ≈ 2 .* (H̃ * x) .- y0
    end

    @testset "map-back applies H to every Krylov vector" begin
        Xr, Yr = randn(T, bs, D), zeros(T, bs, D)
        Polfed._mapback(RowLayout(), H)(Yr, Xr)
        @test Yr ≈ permutedims(Matrix(H) * permutedims(Xr))
        Xc, Yc = randn(T, D, bs), zeros(T, D, bs)
        Polfed._mapback(ColumnLayout(), H)(Yc, Xc)
        @test Yc ≈ Matrix(H) * Xc
    end
end

@testset "value-grouped kernels are refused where they do not exist" begin
    H = sprandn(16, 16, 0.3); H = H + H'
    @test_throws ArgumentError Polfed._build_matrix_ops(H, :auto, 2, 1.0, 0.0, BlockWise(),
                                                        ColumnLayout())
end

@testset "_mkl_axpy!, $T" for T in (Float64, Float32, ComplexF64, ComplexF32)
    bs, D = 3, 40
    Xt, Yt = randn(T, bs, D), randn(T, bs, D)
    want = Yt .+ T(0.3) .* Xt
    @test Polfed._mkl_axpy!(0.3, Xt, Yt) === Yt
    @test Yt ≈ want
end

@testset "_mkl_axpy! falls back for views and non-BLAS types" begin
    bs, D = 3, 40
    X = randn(bs, D)
    big = randn(bs + 2, D)
    Yv = view(big, 2:bs+1, :)
    want = Yv .+ 0.5 .* X
    Polfed._mkl_axpy!(0.5, X, Yv)
    @test Yv ≈ want
    Xr, Yr = rand(1:5, bs, D) .// 3, rand(1:5, bs, D) .// 7
    @test Polfed._mkl_axpy!(2, Xr, copy(Yr)) == Yr .+ 2 .* Xr
end
