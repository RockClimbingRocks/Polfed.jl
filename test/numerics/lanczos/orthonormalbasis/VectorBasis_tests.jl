using Test
using LinearAlgebra
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

@testset "VectorBasis" begin
    pu = CPU()

    @testset "scalar" begin
        @testset "add! within capacity" begin
            b = LZ.createbasis(4, zeros(3), VectorBasis, pu)
            @test b.nvecs == 0
            @test b.blocksize == 1

            LZ.add!(b, [1.0, 1.0, 1.0])
            LZ.add!(b, [2.0, 2.0, 2.0])
            @test b.nvecs == 2
            av = LZ.allvecs(b)
            @test av[1] == [1.0, 1.0, 1.0]
            @test av[2] == [2.0, 2.0, 2.0]
        end

        @testset "add! overflow shifts the oldest entry out" begin
            b = LZ.createbasis(3, zeros(2), VectorBasis, pu)
            for k in 1:3
                LZ.add!(b, Float64[k, k])
            end
            @test b.nvecs == 3

            LZ.add!(b, [4.0, 4.0])           # full → drop oldest entry
            @test b.nvecs == 3
            av = LZ.allvecs(b)
            @test av[1] == [2.0, 2.0]
            @test av[2] == [3.0, 3.0]
            @test av[3] == [4.0, 4.0]
        end

        @testset "allvecs on an empty basis throws" begin
            b = LZ.createbasis(2, zeros(2), VectorBasis, pu)
            @test_throws ArgumentError LZ.allvecs(b)
        end

        @testset "all_withoutlasttwo" begin
            b = LZ.createbasis(5, zeros(2), VectorBasis, pu)
            for k in 1:4
                LZ.add!(b, Float64[k, k])
            end
            # nvecs = 4 → exclude last two entries → entries 1,2
            awl = LZ.all_withoutlasttwo(b)
            @test length(awl) == 2
            @test awl[1] == [1.0, 1.0]
            @test awl[2] == [2.0, 2.0]
        end

        @testset "all_withoutlasttwo guard: too few entries" begin
            b = LZ.createbasis(5, zeros(2), VectorBasis, pu)
            LZ.add!(b, [1.0, 1.0])           # nvecs = 1 < 2
            @test_throws ArgumentError LZ.all_withoutlasttwo(b)

            LZ.add!(b, [2.0, 2.0])           # nvecs = 2 → empty
            @test length(LZ.all_withoutlasttwo(b)) == 0
        end
    end

    @testset "block (blocksize 2)" begin
        blk(a, b) = Float64[a b; a b; a b]   # 3×2 block with constant columns a, b

        @testset "constructor sizes entries by the column budget" begin
            # maxdim is a column budget → cld(6, 2) = 3 entries, each a 3×2 block.
            b = LZ.createbasis(6, zeros(3, 2), VectorBasis, pu)
            @test b.blocksize == 2
            @test b.nvecs == 0
            @test length(b.basis) == 3
        end

        @testset "add! appends whole blocks" begin
            b = LZ.createbasis(6, zeros(3, 2), VectorBasis, pu)
            LZ.add!(b, blk(1, 2))
            LZ.add!(b, blk(3, 4))
            @test b.nvecs == 2
            av = LZ.allvecs(b)
            @test av[1] == blk(1, 2)
            @test av[2] == blk(3, 4)
        end

        @testset "add! overflow shifts the oldest block out" begin
            b = LZ.createbasis(4, zeros(3, 2), VectorBasis, pu)   # cld(4,2) = 2 entries
            LZ.add!(b, blk(1, 2))
            LZ.add!(b, blk(3, 4))                                 # full
            LZ.add!(b, blk(5, 6))                                 # drop oldest block
            @test b.nvecs == 2
            av = LZ.allvecs(b)
            @test av[1] == blk(3, 4)
            @test av[2] == blk(5, 6)
        end

        @testset "all_withoutlasttwo excludes the last two blocks" begin
            b = LZ.createbasis(6, zeros(3, 2), VectorBasis, pu)
            LZ.add!(b, blk(1, 2))
            LZ.add!(b, blk(3, 4))
            LZ.add!(b, blk(5, 6))                                 # nvecs = 3 blocks
            awl = LZ.all_withoutlasttwo(b)                        # exclude last two → first block
            @test length(awl) == 1
            @test awl[1] == blk(1, 2)
        end

        @testset "all_withoutlasttwo guard: fewer than two blocks" begin
            b = LZ.createbasis(6, zeros(3, 2), VectorBasis, pu)
            LZ.add!(b, blk(1, 2))                                 # nvecs = 1 < 2
            @test_throws ArgumentError LZ.all_withoutlasttwo(b)
        end
    end
end
