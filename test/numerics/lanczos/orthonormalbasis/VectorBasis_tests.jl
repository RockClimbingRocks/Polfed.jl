using Test
using LinearAlgebra
using Polfed
using Polfed.Numerics.Lanczos

LZ = Polfed.Numerics.Lanczos

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

        @testset "growth past maxdim keeps every entry" begin
            # `maxdim` is a capacity HINT for this backend, not a limit: the basis grows on
            # demand and never drops a vector (dropping one would leave full reorthogonalization
            # projecting against an incomplete basis).
            b = LZ.createbasis(3, zeros(2), VectorBasis, pu)
            for k in 1:6                     # deliberately twice the hint
                LZ.add!(b, Float64[k, k])
            end
            @test b.nvecs == 6
            av = LZ.allvecs(b)
            @test length(av) == 6
            @test av[1] == [1.0, 1.0]        # oldest still present
            @test av[6] == [6.0, 6.0]
        end

        @testset "no preallocation before the first add!" begin
            b = LZ.createbasis(1000, zeros(2), VectorBasis, pu)
            @test b.nvecs == 0
            @test isempty(b.basis)           # a huge budget costs nothing up front
        end

        @testset "add! copies, never aliases the caller's buffer" begin
            # Lanczos reuses one work buffer across iterations; storing it by reference would
            # make every entry the same vector.
            b = LZ.createbasis(4, zeros(2), VectorBasis, pu)
            work = [1.0, 1.0]
            LZ.add!(b, work)
            work .= [9.0, 9.0]               # caller reuses the buffer
            LZ.add!(b, work)
            av = LZ.allvecs(b)
            @test av[1] == [1.0, 1.0]
            @test av[2] == [9.0, 9.0]
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

        @testset "constructor allocates no entries" begin
            b = LZ.createbasis(6, zeros(3, 2), VectorBasis, pu)
            @test b.blocksize == 2
            @test b.nvecs == 0
            @test isempty(b.basis)           # entries appear only as blocks are added
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

        @testset "growth past maxdim keeps every block" begin
            b = LZ.createbasis(4, zeros(3, 2), VectorBasis, pu)   # hint of cld(4,2) = 2 entries
            LZ.add!(b, blk(1, 2))
            LZ.add!(b, blk(3, 4))
            LZ.add!(b, blk(5, 6))                                 # past the hint — no drop
            @test b.nvecs == 3
            av = LZ.allvecs(b)
            @test av[1] == blk(1, 2)
            @test av[2] == blk(3, 4)
            @test av[3] == blk(5, 6)
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
