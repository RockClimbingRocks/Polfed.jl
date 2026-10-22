# FlipStructure: the term list, its normalisation, and recovering it from a matrix.
#
# The normalisation cases are the point. Each one is a SILENT wrong answer if it is not handled
# at construction — the kernels run under `@inbounds`, so nothing throws at matvec time.

using LinearAlgebra, SparseArrays
import UniPolfed.Optimization: FlipStructure, n_masks, rescale, classify, _derive_flip

"""    sun_sparse(L; α) -> (H, masks, values, diags) — 2-bit masks, all values different."""
function sun_sparse(L::Int; α = 0.75, T = Float64)
    D = 1 << L
    diags = randn(T, D)
    masks = [1 | (1 << j) for j in 1:L-1]
    vals  = T[T(α)^(j - 1) for j in 1:L-1]
    I = Int[]; J = Int[]; V = T[]
    for p in 0:D-1
        push!(I, p+1); push!(J, p+1); push!(V, diags[p+1])
        for (m, v) in zip(masks, vals)
            push!(I, p+1); push!(J, xor(p, m) + 1); push!(V, v)
        end
    end
    return sparse(I, J, V, D, D), masks, vals, diags
end

@testset "FlipStructure" begin

    @testset "normalisation at construction" begin
        d = zeros(8)

        # a zero mask is a DIAGONAL contribution, not an off-diagonal
        fs = FlipStructure(d, [0, 1], [2.0, 0.5])
        @test fs.masks == [1] && fs.values == [0.5]
        @test all(fs.diags .== 2.0)
        @test d == zeros(8)                       # the caller's diagonal is not mutated

        # duplicates merge (summing values) rather than visiting the neighbour twice
        fs = FlipStructure(zeros(8), [3, 1, 3], [1.0, 0.25, 2.0])
        @test fs.masks == [1, 3] && fs.values == [0.25, 3.0]

        # a term that merges to zero is dropped
        @test n_masks(FlipStructure(zeros(8), [2, 2], [1.0, -1.0])) == 0

        # masks come out ascending, so the same operator always builds the same kernel
        @test FlipStructure(zeros(16), [8, 1, 4], ones(3)).masks == [1, 4, 8]

        @test FlipStructure(zeros(8), [1], [1.0]).L == 3
    end

    @testset "rejects what would read out of bounds" begin
        @test_throws ArgumentError FlipStructure(zeros(6), [1], [1.0])       # dim not 2^L
        @test_throws ArgumentError FlipStructure(zeros(8), [1, 2], [1.0])    # length mismatch
        @test_throws ArgumentError FlipStructure(zeros(8), [8], [1.0])       # mask >= dim
        @test_throws ArgumentError FlipStructure(zeros(8), [-1], [1.0])      # negative mask
    end

    @testset "rescale scales values and diagonal, never the masks" begin
        fs = FlipStructure(fill(4.0, 8), [1, 2], [2.0, 6.0])
        r = rescale(fs, 2.0, 1.0)
        @test r.masks == fs.masks                       # rescaling never moves an off-diagonal
        @test r.values ≈ [1.0, 3.0]
        @test all(r.diags .≈ (4.0 - 1.0) / 2.0)
        @test fs.values == [2.0, 6.0]                   # source untouched
    end

    @testset "_derive_flip recovers the term list from a matrix" begin
        for (name, H, masks, vals) in (
                ("sx  (1-bit masks, shared value)",
                 let L = 8, D = 1 << 8, hx = 0.7
                     m = [1 << b for b in 0:L-1]
                     (qrem(L; hx = hx), m, fill(hx, L))
                 end...),
                ("sun (2-bit masks, all different)",
                 let (H, m, v, _) = sun_sparse(8); (H, m, v) end...))
            fs = _derive_flip(classify(H))
            @test !isnothing(fs)
            @test fs.masks == sort(masks)
            # values must follow their mask through the sort
            perm = sortperm(masks)
            @test fs.values ≈ vals[perm]
            @test fs.dim == size(H, 1) && fs.L == 8
        end
    end

    @testset "_derive_flip declines what is not a flip operator" begin
        # a symmetry sector: the index is a rank, not a configuration, and dim is not 2^L
        H = UniPolfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.55, 0.3;
                                             boundary = :periodic, use_sparse = true)
        @test isnothing(_derive_flip(classify(H)))

        # right dimension, wrong structure: move ONE entry off its xor position
        H2 = Matrix(qrem(4))
        H2[1, 4] = 0.0; H2[1, 6] = 0.7
        @test isnothing(_derive_flip(classify(sparse(H2))))

        # right positions, but one entry has the wrong amplitude for its mask
        H3 = Matrix(qrem(4))
        H3[1, 2] = 99.0
        @test isnothing(_derive_flip(classify(sparse(H3))))
    end
end
