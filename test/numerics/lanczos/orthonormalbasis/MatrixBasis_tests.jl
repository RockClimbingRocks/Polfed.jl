using Test
using LinearAlgebra
using Polfed
using Polfed.Numerics.Lanczos

LZ = Polfed.Numerics.Lanczos

@testset "MatrixBasis" begin
    pu = CPU()

    @testset "add! within capacity (vectors)" begin
        b = LZ.createbasis(4, zeros(3), MatrixBasis, pu)
        @test b.nvecs == 0
        @test b.blocksize == 1

        LZ.add!(b, [1.0, 1.0, 1.0])
        LZ.add!(b, [2.0, 2.0, 2.0])
        @test b.nvecs == 2
        @test LZ.allvecs(b) == [1.0 2.0; 1.0 2.0; 1.0 2.0]
    end

    @testset "add! overflow shifts the oldest column out (vectors)" begin
        b = LZ.createbasis(3, zeros(2), MatrixBasis, pu)
        for k in 1:3
            LZ.add!(b, Float64[k, k])
        end
        @test b.nvecs == 3
        @test LZ.allvecs(b) == [1.0 2.0 3.0; 1.0 2.0 3.0]

        LZ.add!(b, [4.0, 4.0])           # full → drop oldest (col 1)
        @test b.nvecs == 3
        @test LZ.allvecs(b) == [2.0 3.0 4.0; 2.0 3.0 4.0]

        LZ.add!(b, [5.0, 5.0])
        @test LZ.allvecs(b) == [3.0 4.0 5.0; 3.0 4.0 5.0]
    end

    @testset "allvecs on an empty basis throws" begin
        b = LZ.createbasis(2, zeros(2), MatrixBasis, pu)
        @test_throws ArgumentError LZ.allvecs(b)
    end

    @testset "all_withoutlasttwo (vectors)" begin
        b = LZ.createbasis(5, zeros(2), MatrixBasis, pu)
        for k in 1:4
            LZ.add!(b, Float64[k, k])
        end
        # nvecs = 4 → exclude last two columns → columns 1,2
        @test LZ.all_withoutlasttwo(b) == [1.0 2.0; 1.0 2.0]
    end

    @testset "all_withoutlasttwo guard: too few vectors" begin
        b = LZ.createbasis(5, zeros(2), MatrixBasis, pu)
        LZ.add!(b, [1.0, 1.0])                         # nvecs = 1 < 2
        @test_throws ArgumentError LZ.all_withoutlasttwo(b)

        LZ.add!(b, [2.0, 2.0])                         # nvecs = 2 → empty view
        @test size(LZ.all_withoutlasttwo(b), 2) == 0
    end

    @testset "block add! and overflow shift (blocksize 2)" begin
        x0 = zeros(3, 2)
        b = LZ.createbasis(4, x0, MatrixBasis, pu)     # maxcols = 4 → 2 blocks
        @test b.blocksize == 2

        blk1 = [1.0 2.0; 1.0 2.0; 1.0 2.0]
        blk2 = [3.0 4.0; 3.0 4.0; 3.0 4.0]
        LZ.add!(b, blk1)
        LZ.add!(b, blk2)
        @test b.nvecs == 4
        @test LZ.allvecs(b) == hcat(blk1, blk2)

        blk3 = [5.0 6.0; 5.0 6.0; 5.0 6.0]
        LZ.add!(b, blk3)                               # 4+2 > 4 → drop oldest block
        @test b.nvecs == 4
        @test LZ.allvecs(b) == hcat(blk2, blk3)
    end

    @testset "block all_withoutlasttwo (blocksize 2)" begin
        b = LZ.createbasis(6, zeros(2, 2), MatrixBasis, pu)
        LZ.add!(b, [1.0 2.0; 1.0 2.0])
        LZ.add!(b, [3.0 4.0; 3.0 4.0])
        LZ.add!(b, [5.0 6.0; 5.0 6.0])                 # nvecs = 6 (3 blocks)
        # exclude last two blocks (4 columns) → first block, columns 1:2
        @test LZ.all_withoutlasttwo(b) == [1.0 2.0; 1.0 2.0]
    end

    @testset "block all_withoutlasttwo guard: fewer than two blocks" begin
        b = LZ.createbasis(6, zeros(2, 2), MatrixBasis, pu)
        LZ.add!(b, [1.0 2.0; 1.0 2.0])                 # nvecs = 2 < 2*blocksize = 4
        @test_throws ArgumentError LZ.all_withoutlasttwo(b)
    end
end
