@testset "value_grouping" begin
    L = 4
    D = 2^L
    H = qrem(L)
    s = value_grouping(H)

    @test s isa MappingStructure
    @test s.dim == D
    @test s.L == L
    @test s.hx ≈ 0.7
    @test length(s.diags) == D
    @test length(s.nb32) == D * L
    @test length(s.flat) == D * L
    @test length(s.starts) == D

    # the diagonal matches H
    @test s.diags ≈ [H[i, i] for i in 1:D]

    # extracted neighbours are exactly the single-bit flips of each site (ELL == CSR)
    for i in 1:D
        ell = Set(Int(s.nb32[(i - 1) * L + c]) for c in 1:L)
        csr = Set(s.flat[s.starts[i] + c] for c in 0:L-1)
        want = Set(xor(i - 1, 1 << b) + 1 for b in 0:L-1)
        @test ell == want
        @test csr == want
    end

    # Stage-1 assumptions are enforced
    @test_throws ArgumentError value_grouping(sparse(Diagonal([1.0, 2.0, 3.0])))                 # no off-diagonals
    @test_throws ArgumentError value_grouping(sparse([1, 2, 3, 4], [2, 1, 4, 3],
                                                     [1.0, 1.0, 2.0, 2.0], 4, 4))                # two off-diagonal values
end
