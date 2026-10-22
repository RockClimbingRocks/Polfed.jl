# The generic row operator of the matrix entrypoints (`optimize_mapping = false`): `op(Y, X)` plus
# the accumulating forms AxpbyClenshaw runs on, checked against plain dense arithmetic.

using Test
using Polfed
using LinearAlgebra, SparseArrays, Random

@testset "generic row operator" begin
    Random.seed!(5)
    D, bs, a, b = 40, 3, 2.5, 0.4
    H = sprandn(D, D, 0.2); H = H + H'
    H̃ = (Matrix(H) - b * I) / a
    op = Polfed._generic_row_rescaled(H, a, b)
    X = randn(bs, D)

    @testset "op(Y, X) = X·H̃" begin
        Y = fill(NaN, bs, D)
        op(Y, X)
        @test Y ≈ X * H̃
    end

    @testset "_row_axpby! accumulates; β = 0 does not read Y" begin
        Y0 = randn(bs, D)
        Y = copy(Y0)
        Polfed._row_axpby!(op, Y, X, 2.0, -1.0)
        @test Y ≈ 2 .* (X * H̃) .- Y0
        Y = fill(NaN, bs, D)
        Polfed._row_axpby!(op, Y, X, 1.5, 0)
        @test Y ≈ 1.5 .* (X * H̃)
    end

    @testset "_mkl_axpy!, $T" for T in (Float64, Float32, ComplexF64, ComplexF32)
        Xt, Yt = randn(T, bs, D), randn(T, bs, D)
        want = Yt .+ T(0.3) .* Xt
        @test Polfed._mkl_axpy!(0.3, Xt, Yt) === Yt
        @test Yt ≈ want
    end

    @testset "_mkl_axpy! falls back for views and non-BLAS types" begin
        big = randn(bs + 2, D)
        Yv = view(big, 2:bs+1, :)
        want = Yv .+ 0.5 .* X
        Polfed._mkl_axpy!(0.5, X, Yv)
        @test Yv ≈ want
        Xr, Yr = rand(1:5, bs, D) .// 3, rand(1:5, bs, D) .// 7
        @test Polfed._mkl_axpy!(2, Xr, copy(Yr)) == Yr .+ 2 .* Xr
    end
end
