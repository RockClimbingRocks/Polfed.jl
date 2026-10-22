# HopStructure: the lattice, its validation, and — the part that can silently lie — recovering
# the lattice from a matrix.
#
# `_derive_hop` reconstructs the shape from row 1 alone and then verifies every row. Both halves
# are tested against ground truth: a lattice it must recover exactly, and near-misses it must
# REFUSE rather than fit a wrong shape to.

using LinearAlgebra, SparseArrays
import Polfed.Optimization: HopStructure, classify, rescale, n_axes, _derive_hop,
                               MappingStructure

"""
    anderson_sparse(shape; t = 0.7, T = Float64, diags = randn(T, prod(shape))) -> (H, diags)

The periodic Anderson operator on a rectangular lattice, assembled the long way — one explicit
`(row, col)` per bond — so it is an independent statement of the operator the kernels compute,
not a re-expression of their index arithmetic. Sites are linearised x-fastest.
"""
function anderson_sparse(shape; t = 0.7, T = Float64, diags = randn(T, prod(shape)))
    D = length(shape)
    strides = [k == 1 ? 1 : prod(shape[1:k-1]) for k in 1:D]
    n = prod(shape)
    I = Int[]; J = Int[]; V = T[]
    for i in 1:n
        p = i - 1
        push!(I, i); push!(J, i); push!(V, T(diags[i]))
        for k in 1:D
            s = strides[k]; L = shape[k]; c = rem(div(p, s), L)
            push!(I, i); push!(J, (c == 0      ? p + (L - 1) * s : p - s) + 1); push!(V, T(t))
            push!(I, i); push!(J, (c == L - 1  ? p - (L - 1) * s : p + s) + 1); push!(V, T(t))
        end
    end
    return sparse(I, J, V, n, n), diags
end

@testset "HopStructure" begin

    @testset "construction and derived fields" begin
        hs = HopStructure(randn(4 * 5 * 6), (4, 5, 6), -1.0)
        @test hs.dim == 120
        @test hs.shape == (4, 5, 6)
        @test hs.strides == (1, 4, 20)           # s₁ = 1, s_{k+1} = s_k·L_k
        @test hs.hopping == -1.0
        @test n_axes(hs) == 3
        @test eltype(typeof(hs)) === Float64
        # the shape may be any iterable; the extents become a static tuple
        @test HopStructure(randn(12), [3, 4], 1.0).shape === (3, 4)
    end

    @testset "degenerate lattices are refused, not silently double-counted" begin
        # a periodic axis of length 2 makes +ê and -ê the same site
        @test_throws ArgumentError HopStructure(randn(8), (2, 4), 1.0)
        @test_throws ArgumentError HopStructure(randn(4), (4, 1), 1.0)
        @test_throws ArgumentError HopStructure(randn(4), (), 1.0)
        @test_throws ArgumentError HopStructure(randn(11), (3, 4), 1.0)   # prod ≠ length
    end

    @testset "rescale is (H - b)/a on the diagonal and the amplitude" begin
        diags = randn(60)
        hs = HopStructure(copy(diags), (3, 4, 5), -1.0)
        a, b = 2.5, 0.4
        hr = rescale(hs, a, b)
        @test hr.diags ≈ (diags .- b) ./ a
        @test hr.hopping ≈ -1.0 / a
        @test hr.shape == hs.shape && hr.strides == hs.strides
        @test hs.diags == diags                  # the original is untouched
    end

    @testset "_derive_hop recovers the lattice — $shape" for shape in
            ((12,), (4, 5), (4, 5, 3), (3, 4, 3, 3), (3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3),
             (3, 3, 3, 3, 3, 3, 3))
        H, diags = anderson_sparse(shape; t = -1.0)
        hs = _derive_hop(classify(H))
        @test !isnothing(hs)
        @test hs.shape == shape
        @test hs.dim == prod(shape)
        @test hs.hopping ≈ -1.0
        @test hs.diags ≈ diags
    end

    @testset "_derive_hop refuses what is not this operator" begin
        # two distinct amplitudes: not one uniform hopping
        H, _ = anderson_sparse((4, 5))
        H[1, 2] = 0.11; H[2, 1] = 0.11
        @test isnothing(_derive_hop(classify(dropzeros!(H))))

        # a bond removed: the row check must catch it even though row 1 still reconstructs
        H2, _ = anderson_sparse((4, 5, 3))
        H2[8, 12] = 0.0; H2[12, 8] = 0.0
        @test isnothing(_derive_hop(classify(dropzeros!(H2))))

        # an OPEN chain — same shape story, but the wrap bonds are missing
        n = 12
        I = Int[]; J = Int[]; V = Float64[]
        for i in 1:n
            push!(I, i); push!(J, i); push!(V, randn())
            i > 1 && (push!(I, i); push!(J, i - 1); push!(V, 0.7))
            i < n && (push!(I, i); push!(J, i + 1); push!(V, 0.7))
        end
        @test isnothing(_derive_hop(classify(sparse(I, J, V, n, n))))

        # a spin-flip operator is emphatically not a lattice
        @test isnothing(_derive_hop(classify(qrem(8))))
    end

    @testset "the derived structure reproduces the matrix it came from" begin
        # the strongest statement: derive, then rebuild the matrix from the recovered lattice
        for shape in ((4, 5, 3), (3, 3, 3, 3))
            H, diags = anderson_sparse(shape; t = -1.0, diags = randn(prod(shape)))
            hs = _derive_hop(classify(H))
            Hback, _ = anderson_sparse(hs.shape; t = hs.hopping, diags = hs.diags)
            @test Hback ≈ H
        end
    end
end
