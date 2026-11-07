# classify: the value-grouped description of a Hamiltonian. Asserted against structures whose
# grouping is known by construction, and — crucially — against the two shapes the previous
# extractor REFUSED: rows of differing length, and several distinct off-diagonal values.

using LinearAlgebra, SparseArrays
using Polfed.Models: xxz_hamiltonian, j1j2_hamiltonian

@testset "MappingStructure / classify" begin

    @testset "single value, every row the same length" begin
        L = 4; D = 2^L
        H = qrem(L)                       # qrem randomizes the diagonal — build it ONCE
        s = classify(H)
        @test s.dim == D
        @test n_values(s) == 1
        @test is_regular(s)
        @test s.lmin == L == s.lmax
        @test s.groups[1].value ≈ 0.7
        @test s.diags ≈ [H[i, i] for i in 1:D]

        # each row's neighbours are exactly the single-bit flips
        g = s.groups[1]
        for i in 1:D
            have = Set(Int(g.flat[c]) for c in g.starts[i]:g.starts[i+1]-1)
            @test have == Set(xor(i - 1, 1 << b) + 1 for b in 0:L-1)
        end
        # starts is a proper CSR offset array of length D+1
        @test length(g.starts) == D + 1
        @test g.starts[end] == length(g.flat) + 1
    end

    @testset "single value, rows of DIFFERING length (used to be rejected)" begin
        H = xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic, field=0.0, use_sparse=true)
        s = classify(H)
        @test n_values(s) == 1
        @test !is_regular(s)
        @test s.lmin < s.lmax
    end

    @testset "several distinct values (used to be rejected)" begin
        H = j1j2_hamiltonian(10, 5, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true)
        s = classify(H)
        @test n_values(s) >= 2
        # groups come back in a deterministic (ascending value) order
        @test issorted([real(g.value) for g in s.groups])
    end

    @testset "grouping is exact: every off-diagonal lands in exactly one group" begin
        H = j1j2_hamiltonian(8, 4, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true)
        s = classify(H)
        D = s.dim
        # rebuild H from the structure and compare
        R = zeros(D, D)
        for i in 1:D
            R[i, i] = s.diags[i]
            for g in s.groups, c in g.starts[i]:g.starts[i+1]-1
                R[i, Int(g.flat[c])] += g.value
            end
        end
        @test maximum(abs.(R .- Matrix(H))) < 1e-12
    end

    @testset "rescale shifts the diagonal and scales the values, sharing the tables" begin
        s = classify(qrem(4))
        a, b = 3.0, 0.5
        r = rescale(s, a, b)
        @test r.diags ≈ (s.diags .- b) ./ a
        @test r.groups[1].value ≈ s.groups[1].value / a
        @test r.groups[1].flat === s.groups[1].flat        # index tables are shared, not copied
        @test r.groups[1].starts === s.groups[1].starts
    end

    @testset "rejected input" begin
        @test_throws ArgumentError classify(sparse([1.0 2.0; 3.0 4.0; 5.0 6.0]))   # not square
        @test_throws ArgumentError classify(sparse(Diagonal([1.0, 2.0, 3.0])))     # no off-diagonals
    end
end
