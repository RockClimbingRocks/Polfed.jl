# spin_flip validation: which operators the pattern claims, and which variant gets chosen.

using LinearAlgebra, SparseArrays
import UniPolfed.Optimization: classify, can_apply, why_not, auto_score, _default_kernel,
                               _packed_lane_width, resolve_mapping_kernel, MappingKernel,
                               build_kernel

@testset "spin_flip / validate" begin

    @testset "claims flip operators, declines everything else" begin
        for H in (qrem(8), sun_sparse(8)[1])
            @test can_apply(Val(:spin_flip), classify(H))
            @test can_apply(Val(:spin_flip), Val(:packed), classify(H))
        end

        # a symmetry sector — the index is a rank, not a spin configuration
        sec = classify(UniPolfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.55, 0.3;
                                                        boundary = :periodic, use_sparse = true))
        @test !can_apply(Val(:spin_flip), sec)
        @test occursin("not 2^L", why_not(Val(:spin_flip), sec))
        # the universal pattern still accepts it — that is the point of the fallback
        @test can_apply(Val(:universal), sec)
    end

    @testset "packed declines complex; rows does not" begin
        c = classify(qrem(7; T = ComplexF64))
        @test can_apply(Val(:spin_flip), c)                          # the PATTERN applies
        @test !can_apply(Val(:spin_flip), Val(:packed), c)           # this VARIANT does not
        @test occursin("Complex", why_not(Val(:spin_flip), Val(:packed), c))
        @test can_apply(Val(:spin_flip), Val(:rows), c)
    end

    @testset "auto_score routes by block size" begin
        W = _packed_lane_width(Float64)
        @test auto_score(Val(:spin_flip), Val(:packed), 1) > 0
        @test auto_score(Val(:spin_flip), Val(:packed), 4) > 0
        @test auto_score(Val(:spin_flip), Val(:packed), 3) < 0        # not a power of two
        @test auto_score(Val(:spin_flip), Val(:packed), 2W) < 0       # wider than a pack
        # rows is always eligible, and outranks the universal kernels
        @test auto_score(Val(:spin_flip), Val(:rows), 3) >
              auto_score(Val(:universal), Val(:row_read), 3)
        @test auto_score(Val(:spin_flip), Val(:packed), 1) >
              auto_score(Val(:spin_flip), Val(:rows), 1)
    end

    @testset "the automatic path stays on universal even where flip kernels apply" begin
        # `optimize_mapping = true` is wired to the universal pattern ONLY. The flip kernels
        # still score higher (asserted above) and still run — they are simply not auto-selected,
        # so a spin-flip matrix has to be asked for by name to get one. This testset guards the
        # wiring, not the scoring.
        s = classify(sun_sparse(8)[1])
        @test _default_kernel(s, 1).pattern === :universal
        @test _default_kernel(s, 4).pattern === :universal
        @test _default_kernel(s, 3).pattern === :universal
        @test _default_kernel(classify(qrem(7; T = ComplexF64)), 1).pattern === :universal

        # `row_bucketed` would score higher at bs = 1, but it needs a SINGLE value group and
        # this operator has several, so `can_apply` rules it out before scoring — the universal
        # default is `row_read` at every block size here.
        @test _default_kernel(s, 1).variant === :row_read
        @test _default_kernel(s, 4).variant === :row_read

        # naming a flip kernel still works — it is un-defaulted, not withdrawn
        @test build_kernel(MappingKernel(:spin_flip, :packed), s, Val(4)) isa Function

        # a non-flip operator lands on universal too, as it always did
        sec = classify(UniPolfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.55, 0.3;
                                                        boundary = :periodic, use_sparse = true))
        @test _default_kernel(sec, 4).pattern === :universal
    end

    @testset "asking for a flip kernel on a non-flip matrix fails legibly" begin
        sec = classify(UniPolfed.Models.xxz_hamiltonian(10, 5, 1.0, 0.55, 0.3;
                                                        boundary = :periodic, use_sparse = true))
        @test_throws ArgumentError resolve_mapping_kernel(:spin_flip_packed, sec, 1)
        @test_throws ArgumentError resolve_mapping_kernel(MappingKernel(:spin_flip, :rows), sec, 1)
    end

    @testset "registered and reported" begin
        names = [(k.pattern, k.variant) for k in UniPolfed.Optimization.REGISTRY]
        @test (:spin_flip, :packed) in names
        @test (:spin_flip, :rows) in names
        out = sprint(io -> UniPolfed.Optimization.mapping_kernels(io, qrem(6)))
        @test occursin("spin_flip_packed", out)
    end
end
