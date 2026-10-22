using Test
using LinearAlgebra
using Random
using Polfed
using Polfed.Numerics.Lanczos

LZ = Polfed.Numerics.Lanczos

# Keep the allocation probe in a function so local Test.@testset bindings do not become
# captured-box allocations in the measurement itself.
_coefficient_reorthogonalization!(Q, W, coeff, pu) =
    Polfed.Numerics.Lanczos.reorthogonalization!(
        Polfed.Numerics.Lanczos.ColumnLayout(), Q, W, coeff, pu)
_coefficient_reorthogonalization_allocations(Q, W, coeff, pu) =
    @allocated _coefficient_reorthogonalization!(Q, W, coeff, pu)

# Low-level Gram-Schmidt kernel: W -= Q (Q' W), removing the component of W in span(Q).
@testset "reorthogonalization! kernel" begin
    pu = CPU()

    @testset "vector becomes orthogonal to the basis" begin
        Random.seed!(2024)
        Q = Matrix(qr(randn(10, 4)).Q)        # orthonormal 10×4 basis
        w = randn(10)
        W = copy(w)
        LZ.reorthogonalization!(LZ.ColumnLayout(),Q, W, pu)
        @test norm(Q' * W) < 1e-12            # orthogonal to every basis column
        @test W ≈ w .- Q * (Q' * w)           # matches the documented projection
    end

    @testset "vector already in the span is annihilated" begin
        Random.seed!(7)
        Q = Matrix(qr(randn(8, 3)).Q)
        W = Q * [2.0, -1.0, 3.0]              # purely in span(Q)
        LZ.reorthogonalization!(LZ.ColumnLayout(),Q, W, pu)
        @test norm(W) < 1e-12
    end

    @testset "vector orthogonal to the span is unchanged" begin
        Q = Matrix{Float64}(I, 5, 2)          # spans e1, e2
        W = [0.0, 0.0, 3.0, 4.0, 0.0]         # orthogonal to e1, e2
        before = copy(W)
        LZ.reorthogonalization!(LZ.ColumnLayout(),Q, W, pu)
        @test W ≈ before
    end

    @testset "block W — every column orthogonal to the basis" begin
        Random.seed!(99)
        Q = Matrix(qr(randn(12, 5)).Q)
        W = randn(12, 3)
        LZ.reorthogonalization!(LZ.ColumnLayout(),Q, W, pu)
        @test norm(Q' * W) < 1e-12
    end

    @testset "complex element type" begin
        Random.seed!(3)
        Q = Matrix(qr(randn(ComplexF64, 8, 3)).Q)
        W = randn(ComplexF64, 8)
        LZ.reorthogonalization!(LZ.ColumnLayout(),Q, W, pu)
        @test norm(Q' * W) < 1e-12
    end

    @testset "reusable coefficient workspace" begin
        @testset "scalar column layout" begin
            Random.seed!(31)
            Q = Matrix(qr(randn(14, 5)).Q)
            w = randn(14)
            W = copy(w)
            coeff = zeros(5)

            LZ.reorthogonalization!(LZ.ColumnLayout(), Q, W, coeff, pu)

            @test W ≈ w - Q * (Q' * w)
            @test coeff ≈ Q' * w
            @test norm(Q' * W) < 1e-12
        end

        @testset "block column layout" begin
            Random.seed!(32)
            Q = Matrix(qr(randn(16, 6)).Q)
            W0 = randn(16, 4)
            W = copy(W0)
            coeff = zeros(6, 4)

            LZ.reorthogonalization!(LZ.ColumnLayout(), Q, W, coeff, pu)

            @test W ≈ W0 - Q * (Q' * W0)
            @test coeff ≈ Q' * W0
            @test norm(Q' * W) < 1e-12
        end

        @testset "block row layout" begin
            Random.seed!(33)
            Q = copy(Matrix(qr(randn(16, 6)).Q)')  # six orthonormal rows, 6×16
            W0 = randn(4, 16)
            W = copy(W0)
            coeff = zeros(4, 6)

            LZ.reorthogonalization!(LZ.RowLayout(), Q, W, coeff, pu)

            @test W ≈ W0 - (W0 * Q') * Q
            @test coeff ≈ W0 * Q'
            @test norm(W * Q') < 1e-12
        end

        @testset "complex block coefficients" begin
            Random.seed!(34)
            Q = Matrix(qr(randn(ComplexF64, 12, 4)).Q)
            W0 = randn(ComplexF64, 12, 3)
            W = copy(W0)
            coeff = zeros(ComplexF64, 4, 3)

            LZ.reorthogonalization!(LZ.ColumnLayout(), Q, W, coeff, pu)

            @test W ≈ W0 - Q * (Q' * W0)
            @test norm(Q' * W) < 1e-12
        end

        @testset "zero allocations after warm-up" begin
            Random.seed!(35)
            Q = Matrix(qr(randn(200, 40)).Q)
            W = randn(200, 8)
            coeff = zeros(40, 8)
            _coefficient_reorthogonalization!(Q, W, coeff, pu)

            bytes = _coefficient_reorthogonalization_allocations(Q, W, coeff, pu)
            @test bytes == 0
        end
    end

    @testset "factorization-owned coefficient workspace" begin
        maxdim = 30

        x = randn(20)
        scalar_basis = LZ.createbasis(maxdim, x, MatrixBasis, pu)
        scalar_fac = LZ.LanczosFactorization(maxdim, scalar_basis, x, pu)
        @test size(scalar_fac.reorth_coeff) == (maxdim,)

        Xcol = randn(20, 4)
        col_basis = LZ.createbasis(maxdim, Xcol, MatrixBasis, pu, LZ.ColumnLayout())
        col_fac = LZ.BlockLanczosFactorization(maxdim, col_basis, Xcol, pu,
                                               LZ.ColumnLayout())
        @test size(col_fac.reorth_coeff) == (maxdim, 4)

        Xrow = randn(4, 20)
        row_basis = LZ.createbasis(maxdim, Xrow, MatrixBasis, pu, LZ.RowLayout())
        row_fac = LZ.BlockLanczosFactorization(maxdim, row_basis, Xrow, pu,
                                               LZ.RowLayout())
        @test size(row_fac.reorth_coeff) == (4, maxdim)

        vector_basis = LZ.createbasis(maxdim, x, VectorBasis, pu)
        vector_fac = LZ.LanczosFactorization(maxdim, vector_basis, x, pu)
        @test vector_fac.reorth_coeff === nothing
    end

    @testset "vector-of-arrays kernel (VectorBasis backend)" begin
        @testset "scalar entries (vector of vectors)" begin
            Random.seed!(11)
            Q = Matrix(qr(randn(10, 4)).Q)
            entries = [Q[:, j] for j in 1:4]      # mutually orthonormal vectors
            w = randn(10); W = copy(w)
            LZ.reorthogonalization!(LZ.ColumnLayout(),entries, W, pu)
            @test norm(Q' * W) < 1e-12
        end

        @testset "block entries (vector of blocks)" begin
            Random.seed!(17)
            Q = Matrix(qr(randn(12, 4)).Q)        # one orthonormal span, split into blocks
            blocks = [Q[:, 1:2], Q[:, 3:4]]       # mutually orthonormal blocks
            W = randn(12, 2)
            LZ.reorthogonalization!(LZ.ColumnLayout(),blocks, W, pu)
            @test norm(Q' * W) < 1e-12
        end
    end
end
