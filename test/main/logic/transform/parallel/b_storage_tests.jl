using Test
using Polfed

buffer_shapes(b) = map(size, b)

@testset "get_b_storage" begin
    pu = Polfed.CPU()

    @testset "BlockWise allocates whole-vector triplet" begin
        x0 = zeros(Float64, 5)
        b = Polfed.get_b_storage(Polfed.BlockWise(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "BlockWise allocates whole-block triplet" begin
        x0 = zeros(Float64, 5, 2)
        b = Polfed.get_b_storage(Polfed.BlockWise(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "VectorWise allocates whole-vector triplet for vector input" begin
        x0 = zeros(Float64, 5)
        b = Polfed.get_b_storage(Polfed.VectorWise(), pu, x0)

        @test length(b) == 3
        @test buffer_shapes(b) == [size(x0), size(x0), size(x0)]
        @test all(eltype(buf) === Float64 for buf in b)
    end

    @testset "VectorWise allocates per-vector triplets" begin
        x0 = zeros(Float64, 5, 2)
        b = Polfed.get_b_storage(Polfed.VectorWise(), pu, x0)

        @test length(b) == size(x0, 2)
        for column_b in b
            @test length(column_b) == 3
            @test buffer_shapes(column_b) == [(size(x0, 1),), (size(x0, 1),), (size(x0, 1),)]
            @test all(eltype(buf) === Float64 for buf in column_b)
        end
    end

    @testset "VectorWise buffer SHAPE follows the layout" begin
        # The workspace goes to the same operator as the slice, so it must have the same shape.
        # Column operators are `mul!(y, H, x)` and take vectors; row operators are
        # `mul!(Y, X, H̃)` over a `bs x D` block, and a one-vector block is still 2-D. Handing a
        # length-D vector to a row operator is not slow, it is a MethodError.
        col = zeros(Float64, 5, 2)                     # ColumnLayout: D=5, bs=2
        bc = Polfed.get_b_storage(Polfed.VectorWise(), pu, col, Polfed.ColumnLayout())
        @test length(bc) == 2
        @test all(buf isa AbstractVector && size(buf) == (5,) for t in bc for buf in t)

        row = zeros(Float64, 2, 5)                     # RowLayout: bs=2, D=5 — same problem
        br = Polfed.get_b_storage(Polfed.VectorWise(), pu, row, Polfed.RowLayout())
        @test length(br) == 2
        @test all(buf isa AbstractMatrix && size(buf) == (1, 5) for t in br for buf in t)
    end

    @testset "element type is preserved" begin
        for T in (Float32, ComplexF64)
            x0 = zeros(T, 5, 2)
            b = Polfed.get_b_storage(Polfed.VectorWise(), pu, x0)

            @test all(all(eltype(buf) === T for buf in column_b) for column_b in b)
        end
    end
end
