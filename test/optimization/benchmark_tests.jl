@testset "benchmark_mappings + list_mappings" begin
    L = 6
    H = qrem(L)
    r = benchmark_mappings(H; bs_list = (1, 2), G_list = (1, 2), secs = 0.02)

    @test r isa MappingBench
    @test r.dim == 2^L
    @test r.L == L
    # every applicable kernel (all reads here) passed its correctness check
    @test all(row.ok for row in r.table if row.applicable && row.kernel !== :mul)
    @test haskey(r.best, 1)
    @test r.fastest.kernel in (:csr_simd, :ell32_svec, :csr64_svec)

    # show + list_mappings render without error
    io = IOBuffer()
    show(io, MIME"text/plain"(), r)
    @test occursin("per column", String(take!(io)))

    io = IOBuffer()
    list_mappings(io)
    out = String(take!(io))
    @test occursin("csr_simd", out)
    @test occursin("shuf", out)
end
