using Test
using UniPolfed

buffer_shapes(b) = map(size, b)

@testset "get_b_storage" begin
    pu = UniPolfed.CPU()

    @testset "BlockWise allocates whole-vector triplet" begin
        x0 = zeros(Float64, 5)
        b = UniPolfed.get_b_storage(UniPolfed.BlockWise(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "BlockWise allocates whole-block triplet" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.BlockWise(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "VectorWise allocates whole-vector triplet for vector input" begin
        x0 = zeros(Float64, 5)
        b = UniPolfed.get_b_storage(UniPolfed.VectorWise(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "VectorWise allocates per-vector triplets" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.VectorWise(), pu, x0)

        @test length(b) == size(x0, 2)
        for column_b in b
            @test length(column_b) == 3
            @test buffer_shapes(column_b) == [(size(x0, 1),), (size(x0, 1),), (size(x0, 1),)]
            @test all(eltype(buf) === Float64 for buf in column_b)
        end
    end

    @testset "element type is preserved" begin
        for T in (Float32, ComplexF64)
            x0 = zeros(T, 5, 2)
            b = UniPolfed.get_b_storage(UniPolfed.VectorWise(), pu, x0)

            @test all(all(eltype(buf) === T for buf in column_b) for column_b in b)
        end
    end
end
