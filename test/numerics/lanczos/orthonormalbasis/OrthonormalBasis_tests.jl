using Test
using LinearAlgebra
using Polfed
using Polfed.Numerics.Lanczos

LZ = Polfed.Numerics.Lanczos

@testset "createbasis dispatch" begin
    pu = CPU()

    @testset "selects the requested concrete basis type" begin
        @test LZ.createbasis(4, zeros(3), MatrixBasis, pu) isa MatrixBasis
        @test LZ.createbasis(4, zeros(3), VectorBasis, pu) isa VectorBasis
    end

    @testset "preserves element type and shape" begin
        bf = LZ.createbasis(5, zeros(Float64, 3), MatrixBasis, pu)
        @test eltype(bf.basis) == Float64
        @test size(bf.basis) == (3, 5)

        bc = LZ.createbasis(5, zeros(ComplexF64, 3), MatrixBasis, pu)
        @test eltype(bc.basis) == ComplexF64
    end

    @testset "records the block size from the seed" begin
        @test LZ.createbasis(4, zeros(3), MatrixBasis, pu).blocksize == 1
        @test LZ.createbasis(4, zeros(3, 2), MatrixBasis, pu).blocksize == 2
        @test LZ.createbasis(4, zeros(3, 3), MatrixBasis, pu).blocksize == 3
    end

    @testset "maxdim means different things per backend" begin
        # MatrixBasis PREALLOCATES maxdim columns: the budget is paid up front and is a hard
        # capacity.
        @test size(LZ.createbasis(8, zeros(5, 2), MatrixBasis, pu).basis, 2) == 8
        # VectorBasis grows on demand: maxdim is only a capacity hint, so nothing is allocated
        # and an enormous budget is free.
        @test isempty(LZ.createbasis(8, zeros(5, 2), VectorBasis, pu).basis)
        @test isempty(LZ.createbasis(10^7, zeros(5), VectorBasis, pu).basis)
    end
end

# MatrixBasis stores columns; VectorBasis stores whole blocks as entries. Driven by the
# same add! sequence they must hold the SAME logical columns. Materialize both to a
# common column-matrix form (backend-independent) and compare.
@testset "MatrixBasis vs VectorBasis consistency" begin
    pu = CPU()

    as_cols(x::AbstractMatrix) = Matrix(x)        # MatrixBasis accessor result
    as_cols(x::AbstractVector) = reduce(hcat, x)  # VectorBasis vector-of-arrays

    # Build both backends, run the identical add! sequence, return the two bases.
    function both(maxdim, x0, items)
        mb = LZ.createbasis(maxdim, x0, MatrixBasis, pu)
        vb = LZ.createbasis(maxdim, x0, VectorBasis, pu)
        for it in items
            LZ.add!(mb, it)
            LZ.add!(vb, it)
        end
        return mb, vb
    end

    @testset "scalar — allvecs match within capacity" begin
        items = [Float64[i, 2i, -i] for i in 1:5]
        mb, vb = both(5, zeros(3), items)
        @test as_cols(LZ.allvecs(mb)) == as_cols(LZ.allvecs(vb)) == reduce(hcat, items)
    end

    @testset "scalar — past capacity the backends DIVERGE by design" begin
        items = [Float64[i, i] for i in 1:6]
        mb, vb = both(3, zeros(2), items)
        # MatrixBasis is fixed-size: it ring-buffers, keeping only the newest 3.
        @test as_cols(LZ.allvecs(mb)) == reduce(hcat, items[end-2:end])
        # VectorBasis grows instead, so it still holds all 6 — the point of the backend.
        @test as_cols(LZ.allvecs(vb)) == reduce(hcat, items)
    end

    @testset "block — allvecs match within capacity" begin
        blk(a) = Float64[a a+1; a a+1; a a+1]
        items = [blk(10k) for k in 1:3]
        mb, vb = both(6, zeros(3, 2), items)
        @test as_cols(LZ.allvecs(mb)) == as_cols(LZ.allvecs(vb)) == reduce(hcat, items)
    end

    @testset "block — past capacity the backends DIVERGE by design" begin
        blk(a) = Float64[a a+1; a a+1]
        items = [blk(10k) for k in 1:3]
        mb, vb = both(4, zeros(2, 2), items)                 # MatrixBasis capacity = 2 blocks
        @test as_cols(LZ.allvecs(mb)) == reduce(hcat, items[2:3])   # oldest block dropped
        @test as_cols(LZ.allvecs(vb)) == reduce(hcat, items)        # grown, nothing dropped
    end

    @testset "all_withoutlasttwo matches (scalar)" begin
        items = [Float64[i, -i] for i in 1:4]
        mb, vb = both(5, zeros(2), items)
        @test as_cols(LZ.all_withoutlasttwo(mb)) == as_cols(LZ.all_withoutlasttwo(vb))
        @test as_cols(LZ.all_withoutlasttwo(mb)) == reduce(hcat, items[1:2])
    end

    @testset "all_withoutlasttwo matches (block)" begin
        blk(a) = Float64[a a+1; a a+1; a a+1]
        items = [blk(10k) for k in 1:3]
        mb, vb = both(6, zeros(3, 2), items)
        @test as_cols(LZ.all_withoutlasttwo(mb)) == as_cols(LZ.all_withoutlasttwo(vb))
        @test as_cols(LZ.all_withoutlasttwo(mb)) == items[1]   # exclude last two blocks
    end

    @testset "complex element type stays consistent" begin
        items = [ComplexF64[i + im, 2i - im] for i in 1:4]
        mb, vb = both(4, zeros(ComplexF64, 2), items)
        @test as_cols(LZ.allvecs(mb)) == as_cols(LZ.allvecs(vb)) == reduce(hcat, items)
    end
end
