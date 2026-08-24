# hopping validation: who may run, and who gets CHOSEN.
#
# Two different questions, and the failure modes differ. `can_apply` saying yes to a non-lattice
# would produce a wrong matvec; `auto_score` choosing badly only produces a slow one. Both are
# pinned here, along with the registry wiring that makes the pattern reachable by name.

using LinearAlgebra, SparseArrays
import UniPolfed.Optimization: classify, can_apply, why_not, auto_score, MappingKernel,
                               REGISTRY, _find, resolve_mapping_kernel, build_optimized_mapping,
                               kernel_name, _hop_lane_width, _hop_packed_ok, HopStructure

@testset "hopping validation" begin

    @testset "can_apply accepts a lattice and refuses everything else" begin
        for shape in ((12,), (4, 5), (4, 5, 3), (3, 3, 3, 3, 3, 3, 3))
            s = classify(anderson_sparse(shape; t = -1.0)[1])
            @test can_apply(Val(:hopping), s)
            @test can_apply(Val(:hopping), Val(:rows), s)
        end
        # a spin-flip operator at L = 8 is not a lattice
        s = classify(qrem(8))
        @test !can_apply(Val(:hopping), s)
        @test occursin("lattice", why_not(Val(:hopping), s))
        # several off-diagonal values
        H, _ = anderson_sparse((4, 5)); H[1, 2] = 0.11; H[2, 1] = 0.11
        s2 = classify(dropzeros!(H))
        @test !can_apply(Val(:hopping), s2)
        @test occursin("uniform hopping", why_not(Val(:hopping), s2))
    end

    @testset "packed declines complex and odd x-extents" begin
        Hc, _ = anderson_sparse((6, 5); t = ComplexF64(-1.0), T = ComplexF64,
                                diags = rand(ComplexF64, 30))
        sc = classify(Hc)
        @test !can_apply(Val(:hopping), Val(:packed), sc)
        @test can_apply(Val(:hopping), Val(:rows), sc)
        @test occursin("Complex", why_not(Val(:hopping), Val(:packed), sc))

        # L₁ odd forces P = 1, where packed IS rows but with extra restrictions
        so = classify(anderson_sparse((9, 4); t = -1.0)[1])
        @test !can_apply(Val(:hopping), Val(:packed), so)
        @test can_apply(Val(:hopping), Val(:rows), so)
        @test occursin("odd", why_not(Val(:hopping), Val(:packed), so))

        # even L₁ is fine even when no wide pack divides it
        @test can_apply(Val(:hopping), Val(:packed), classify(anderson_sparse((6, 4); t = -1.0)[1]))
    end

    @testset "auto_score prefers packed exactly where it is expressible" begin
        W = _hop_lane_width(Float64)
        @test auto_score(Val(:hopping), Val(:packed), 1) > auto_score(Val(:hopping), Val(:rows), 1)
        @test auto_score(Val(:hopping), Val(:packed), 3) < 0        # 3 ∤ W
        @test auto_score(Val(:hopping), Val(:packed), 2W) < 0       # wider than a pack
        @test auto_score(Val(:hopping), Val(:rows), 3) > 0
        # both beat the universal fallback, which reads an index table
        @test auto_score(Val(:hopping), Val(:rows), 1) >
              auto_score(Val(:universal), Val(:row_read), 1)
    end

    @testset "_hop_packed_ok tracks the pack width, not a hardcoded size" begin
        W = _hop_lane_width(Float64)
        @test _hop_packed_ok(Float64, 1)
        @test _hop_packed_ok(Float64, W)
        @test !_hop_packed_ok(Float64, W + 1)
        @test !_hop_packed_ok(ComplexF64, 1)
    end

    @testset "registered and reachable by name" begin
        @test !isnothing(_find(:hopping, :packed))
        @test !isnothing(_find(:hopping, :rows))
        @test _find(:hopping, :packed).params == (W = 0, tile = 0)
        @test _find(:hopping, :rows).params == (tile = 0,)
        s = classify(anderson_sparse((8, 5); t = -1.0)[1])
        @test kernel_name(resolve_mapping_kernel(:hopping, s, 1)) === :hopping_packed
        @test kernel_name(resolve_mapping_kernel(:hopping_rows, s, 1)) === :hopping_rows
        # a parameter the kernel does not take is refused at construction
        @test_throws ArgumentError MappingKernel(:hopping, :rows; W = 8)
        @test_throws ArgumentError MappingKernel(:hopping, :packed; G = 4)
    end

    @testset "the automatic path stays on universal even where hopping applies" begin
        # `optimize_mapping = true` is wired to the universal pattern ONLY. The hopping kernels
        # still score higher and still run — they are simply not auto-selected, so a lattice
        # operator has to be asked for by name to get one.
        H, _ = anderson_sparse((8, 5, 3); t = -1.0)
        for bs in (1, 2, 4)
            r = build_optimized_mapping(H, true, Val(bs), 2.5, 0.3)
            @test r.kernel.pattern === :universal
        end

        # naming it still works — un-defaulted, not withdrawn — and still verifies against mul!
        r = build_optimized_mapping(H, MappingKernel(:hopping, :packed), Val(2), 2.5, 0.3)
        @test r.kernel.pattern === :hopping && r.kernel.variant === :packed

        Hc, _ = anderson_sparse((8, 5); t = ComplexF64(-1.0), T = ComplexF64,
                                diags = rand(ComplexF64, 40))
        r = build_optimized_mapping(Hc, MappingKernel(:hopping, :rows), Val(2), 2.5, 0.3)
        @test r.kernel.pattern === :hopping && r.kernel.variant === :rows
    end
end
