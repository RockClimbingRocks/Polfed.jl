# universal/row_read: correctness against `mul!` on every structure classify can produce.
# This is the whole point of the rewrite — the single-value/equal-length case still works, and the
# two shapes that previously errored now work too.

using LinearAlgebra, SparseArrays
using Polfed.Models: xxz_hamiltonian, j1j2_hamiltonian

@testset "universal / row_read" begin

    # (name, H) covering every pattern
    cases = (
        ("equal row length", qrem(6)),
        ("differing lengths", xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic,
                                              field=0.0, use_sparse=true)),
        ("several values",   j1j2_hamiltonian(10, 5, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true)),
    )

    @testset "matches mul! — $name" for (name, H) in cases
        s = classify(H)
        D = s.dim
        for bs in (1, 2, 4)
            f! = build_kernel(MappingKernel(:universal), s, Val(bs))
            X = rand(bs, D); Y = similar(X)
            f!(Y, X)
            @test maximum(abs.(Y .- permutedims(H * permutedims(X)))) < 1e-12
        end
    end

    @testset "matches mul! on the rescaled operator" begin
        H = xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic, field=0.0, use_sparse=true)
        s = classify(H); D = s.dim
        a, b = 4.0, 0.25
        f! = build_kernel(MappingKernel(:universal), rescale(s, a, b), Val(2))
        X = rand(2, D); Y = similar(X)
        f!(Y, X)
        want = permutedims((H * permutedims(X) .- b .* permutedims(X)) ./ a)
        @test maximum(abs.(Y .- want)) < 1e-12
    end

    @testset "complex Hermitian" begin
        H = qrem(5; T = ComplexF64)
        s = classify(H); D = s.dim
        f! = build_kernel(MappingKernel(:universal), s, Val(2))
        X = rand(ComplexF64, 2, D); Y = similar(X)
        f!(Y, X)
        @test maximum(abs.(Y .- permutedims(H * permutedims(X)))) < 1e-12
    end

    @testset "row_bucketed agrees with row_read row for row" begin
        # Bucketing permutes the order rows are VISITED; the result must be identical. Row lengths
        # here span 2…10, so several buckets and the unspecialized tail are all exercised.
        for (name, H) in (("differing lengths", xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0;
                                                boundary=:periodic, field=0.0, use_sparse=true)),
                          ("equal row length", qrem(6)))
            s = classify(H); D = s.dim
            for bs in (1, 2, 4)
                X = rand(bs, D)
                Yr = similar(X); Yb = similar(X)
                build_kernel(MappingKernel(:universal, :row_read),     s, Val(bs))(Yr, X)
                build_kernel(MappingKernel(:universal, :row_bucketed), s, Val(bs))(Yb, X)
                @test Yr == Yb                     # same operations, same order per row: exact
                @test maximum(abs.(Yb .- permutedims(H * permutedims(X)))) < 1e-12
            end
        end
    end

    @testset "row_bucketed covers every row exactly once" begin
        # The failure mode bucketing invites is a row that lands in no bucket (silently zero) or
        # in two (silently doubled). Applying to a basis vector makes either visible.
        H = xxz_hamiltonian(10, 5, 1.0, 0.55, 0.3; boundary=:periodic, use_sparse=true)
        s = classify(H); D = s.dim
        X = ones(1, D); Y = similar(X)
        build_kernel(MappingKernel(:universal, :row_bucketed), s, Val(1))(Y, X)
        @test maximum(abs.(vec(Y) .- vec(sum(H, dims = 2)))) < 1e-12
    end
end
