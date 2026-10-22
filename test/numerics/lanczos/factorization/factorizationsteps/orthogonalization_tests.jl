using Test
using LinearAlgebra
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

# Low-level orthogonalization kernel: r -= (α·v1 + β·v2) for the three-term recurrence.
@testset "orthogonalization! kernels" begin

    @testset "vector kernel — real, ground truth" begin
        r  = [1.0, 2.0, 3.0]
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        LZ.orthogonalization!(LZ.ColumnLayout(),r, v1, v2, 2.0, 3.0)
        # r - (2·v1 + 3·v2) = [1,2,3] - [2,3,0]
        @test r == [-1.0, -1.0, 3.0]
    end

    @testset "vector kernel — complex r, real coefficients" begin
        r  = ComplexF64[1+1im, 2+2im]
        v1 = ComplexF64[1, 0]
        v2 = ComplexF64[0, 1]
        LZ.orthogonalization!(LZ.ColumnLayout(),r, v1, v2, 1.0, 2.0)
        # [1+1im, 2+2im] - (1·[1,0] + 2·[0,1]) = [0+1im, 0+2im]
        @test r ≈ ComplexF64[0+1im, 0+2im]
    end

    @testset "vector kernel — β = 0 means no v2 contribution (first step)" begin
        r  = [5.0, 7.0]
        v1 = [1.0, 1.0]
        v2 = [9.0, 9.0]      # should be ignored when β = 0
        LZ.orthogonalization!(LZ.ColumnLayout(),r, v1, v2, 2.0, 0.0)
        @test r == [3.0, 5.0]
    end

    @testset "block kernel — ground truth" begin
        r  = [4.0 0.0; 0.0 9.0]
        v1 = Matrix{Float64}(I, 2, 2)
        v2 = Matrix{Float64}(I, 2, 2)
        α  = [1.0 0.0; 0.0 2.0]
        β  = [3.0 0.0; 0.0 4.0]
        # r - (v1·α + v2·β') = [4 0;0 9] - ([1 0;0 2] + [3 0;0 4]) = [0 0;0 3]
        LZ.orthogonalization!(LZ.ColumnLayout(),r, v1, v2, α, β)
        @test r == [0.0 0.0; 0.0 3.0]
    end

    @testset "block kernel — β' (adjoint) is used, not β" begin
        r  = zeros(2, 2)
        v1 = zeros(2, 2)
        v2 = Matrix{Float64}(I, 2, 2)
        α  = zeros(2, 2)
        β  = [0.0 5.0; 0.0 0.0]      # β' = [0 0; 5 0]
        LZ.orthogonalization!(LZ.ColumnLayout(),r, v1, v2, α, β)
        # r -= v2 * β' = I * [0 0; 5 0] = [0 0; 5 0]
        @test r == [0.0 0.0; -5.0 0.0]
    end

    @testset "mixed vector-basis / block-residual overload" begin
        # v1, v2 stored as vectors but r is a (dim × 1) block
        r  = reshape([1.0, 2.0, 3.0], 3, 1)
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        α  = reshape([2.0], 1, 1)
        β  = reshape([3.0], 1, 1)
        LZ.orthogonalization!(LZ.ColumnLayout(),r, v1, v2, α, β)
        @test vec(r) == [-1.0, -1.0, 3.0]
    end
end
