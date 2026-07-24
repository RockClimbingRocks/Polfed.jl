using Test
using LinearAlgebra
using Random
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

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
