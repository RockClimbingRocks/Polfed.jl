using Test
using UniPolfed

buffer_shapes(b) = map(size, b)

@testset "get_b_storage" begin
    pu = UniPolfed.CPU()

    @testset "NoParallel allocates whole-vector triplet" begin
        x0 = zeros(Float64, 5)
        b = UniPolfed.get_b_storage(UniPolfed.NoParallel(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "NoParallel allocates whole-block triplet" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.NoParallel(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "MulRowsParallel allocates whole-block triplet" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.MulRowsParallel(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "MulColsParallel allocates whole-vector triplet for vector input" begin
        x0 = zeros(Float64, 5)
        b = UniPolfed.get_b_storage(UniPolfed.MulColsParallel(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "MulColsParallel with nt_per_col > 1 allocates whole-block triplet" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.MulColsParallel(2), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "MulColsParallel with nt_per_col = 1 allocates per-column triplets" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.MulColsParallel(1), pu, x0)

        @test length(b) == size(x0, 2)
        for column_b in b
            @test length(column_b) == 3
            @test buffer_shapes(column_b) == [(size(x0, 1),), (size(x0, 1),), (size(x0, 1),)]
            @test all(eltype(buf) === Float64 for buf in column_b)
        end
    end

    @testset "TwoLevelParallel allocates per-column triplets" begin
        x0 = zeros(Float64, 5, 2)
        b = UniPolfed.get_b_storage(UniPolfed.TwoLevelParallel(1), pu, x0)

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
            b = UniPolfed.get_b_storage(UniPolfed.MulColsParallel(1), pu, x0)

            @test all(all(eltype(buf) === T for buf in column_b) for column_b in b)
        end
    end
end
