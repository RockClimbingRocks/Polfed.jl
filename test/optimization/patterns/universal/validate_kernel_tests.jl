# universal — validation. `can_apply` is the cheap gate before building; `verify_kernel` is the
# check that the built closure actually agrees with `mul!`, which runs on every real build.

using LinearAlgebra, SparseArrays
using Polfed.Models: xxz_hamiltonian, j1j2_hamiltonian
using Polfed.Optimization: auto_score

@testset "universal / validate" begin

    @testset "row_bucketed declines multi-value structures" begin
        one = classify(xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic,
                                       field=0.0, use_sparse=true))
        two = classify(j1j2_hamiltonian(10, 5, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true))
        @test can_apply(Val(:universal), Val(:row_bucketed), one)
        @test !can_apply(Val(:universal), Val(:row_bucketed), two)
        @test occursin("single off-diagonal value",
                       why_not(Val(:universal), Val(:row_bucketed), two))
        # the pattern itself still accepts both — the restriction is the VARIANT's
        @test can_apply(Val(:universal), two)
    end

    @testset "auto_score keeps row_bucketed out of the block-size cases it loses" begin
        @test auto_score(Val(:universal), Val(:row_bucketed), 1) > 0
        @test auto_score(Val(:universal), Val(:row_bucketed), 4) < 0
        @test auto_score(Val(:universal), Val(:row_read), 1) == 0
    end

    @testset "can_apply accepts every structure classify can produce" begin
        for H in (qrem(5),
                  xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic, field=0.0, use_sparse=true),
                  j1j2_hamiltonian(10, 5, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true))
            @test can_apply(Val(:universal), classify(H))
            # variant form falls back to the pattern answer when a kernel adds no restriction
            @test can_apply(Val(:universal), Val(:row_read), classify(H))
        end
        @test why_not(Val(:universal), classify(qrem(4))) isa String
    end

    @testset "verify_kernel passes a correctly built kernel" begin
        H = xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic, field=0.0, use_sparse=true)
        s = classify(H)
        a, b = 4.0, 0.25
        for bs in (1, 2, 4)
            f! = build_kernel(MappingKernel(:universal), rescale(s, a, b), Val(bs))
            @test verify_kernel(f!, H, a, b, Val(bs), :universal_row_read)
        end
    end

    @testset "verify_kernel CATCHES a wrong kernel" begin
        # A checker that never fires is not a checker. Three ways to be wrong, all of which would
        # otherwise return a plausible-looking matvec and silently corrupt the eigenpairs.
        H = qrem(5)
        s = classify(H)
        a, b = 3.0, 0.5
        good = build_kernel(MappingKernel(:universal), rescale(s, a, b), Val(2))

        # forgot to rescale
        unrescaled = build_kernel(MappingKernel(:universal), s, Val(2))
        @test_throws ErrorException verify_kernel(unrescaled, H, a, b, Val(2), :wrong_rescale)

        # right shape, slightly wrong values
        perturbed = (Y, X) -> (good(Y, X); Y .*= 1.001; nothing)
        @test_throws ErrorException verify_kernel(perturbed, H, a, b, Val(2), :perturbed)

        # drops the off-diagonals entirely
        diagonly = (Y, X) -> (Y .= 0; nothing)
        @test_throws ErrorException verify_kernel(diagonly, H, a, b, Val(2), :zeroed)
    end

    @testset "the error says which kernel and by how much" begin
        H = qrem(4)
        bad = (Y, X) -> (Y .= 0; nothing)
        msg = try
            verify_kernel(bad, H, 1.0, 0.0, Val(1), :my_broken_kernel)
        catch e
            sprint(showerror, e)
        end
        @test occursin("my_broken_kernel", msg)
        @test occursin("disagrees with mul!", msg)
    end

    @testset "every build_optimized_mapping verifies" begin
        # the guarantee: the pipeline seam never returns an unverified closure
        H = j1j2_hamiltonian(8, 4, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true)
        nt = build_optimized_mapping(H, true, Val(2), 4.0, 0.25)
        @test nt.kernel isa MappingKernel
        X = rand(2, size(H, 1)); Y = similar(X)
        nt.f!_rescaled(Y, X)
        want = permutedims((H * permutedims(X) .- 0.25 .* permutedims(X)) ./ 4.0)
        @test maximum(abs.(Y .- want)) < 1e-12
    end
end
