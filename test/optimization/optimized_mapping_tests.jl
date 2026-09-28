# resolve_mapping_kernel + build_optimized_mapping: the boundary the POLFED pipeline consumes.

using LinearAlgebra, SparseArrays
using UniPolfed.Models: xxz_hamiltonian, j1j2_hamiltonian

@testset "resolve + build_optimized_mapping" begin
    H = xxz_hamiltonian(10, 5, 1.0, 0.55, 0.0; boundary=:periodic, field=0.0, use_sparse=true)
    s = classify(H)

    @testset "every accepted spelling resolves to the same kernel" begin
        for choice in (true, :auto, :universal, :universal_row_read,
                       MappingKernel(:universal), MappingKernel(:universal, :row_read))
            k = resolve_mapping_kernel(choice, s, 4)
            @test k.pattern === :universal && k.variant === :row_read
        end
        # named-tuple spelling still works
        @test resolve_mapping_kernel((kernel = :universal,), s, 4).variant === :row_read
    end

    @testset "auto-selection depends on the block size" begin
        # H here is single-value, so row_bucketed applies — but only earns selection at bs = 1
        @test resolve_mapping_kernel(true, s, 1).variant === :row_bucketed
        @test resolve_mapping_kernel(true, s, 2).variant === :row_read
        @test resolve_mapping_kernel(true, s, 4).variant === :row_read
        # asking by name overrides the preference at any block size
        @test resolve_mapping_kernel(:universal_row_bucketed, s, 8).variant === :row_bucketed
    end

    @testset "rejected requests" begin
        @test_throws ArgumentError resolve_mapping_kernel(false, s, 1)   # generic path, not here
        @test_throws ArgumentError resolve_mapping_kernel(:no_such, s, 1)
        @test_throws ArgumentError resolve_mapping_kernel(3.14, s, 1)
        # row_bucketed cannot run on a multi-value structure, even asked for by name
        s2 = classify(j1j2_hamiltonian(10, 5, 1.0, 0.5, 1.0, 1.0, 0.0; use_sparse=true))
        @test_throws ArgumentError resolve_mapping_kernel(:universal_row_bucketed, s2, 1)
    end

    @testset "build_optimized_mapping returns a working rescaled operator" begin
        a, b = 4.0, 0.25
        nt = build_optimized_mapping(H, true, Val(2), a, b)
        @test nt.kernel.variant === :row_read

        D = size(H, 1)
        X = rand(2, D); Y = similar(X)
        nt.f!_rescaled(Y, X)
        want = permutedims((H * permutedims(X) .- b .* permutedims(X)) ./ a)
        @test maximum(abs.(Y .- want)) < 1e-12
    end

    @testset "a kernel with a range form carries it, verified" begin
        a, b = 4.0, 0.25
        D = size(H, 1)
        op = build_optimized_mapping(H, true, Val(2), a, b).f!_rescaled
        @test op isa UniPolfed.Optimization.RangeKernelOperator
        @test (op.units, op.alignment) == (D, 1)
        X = rand(2, D); Y1 = similar(X); Y2 = similar(X)
        op(Y1, X)                                             # the whole-block form
        op.range!(Y2, X, 1, 17); op.range!(Y2, X, 18, D)      # the range form, in two chunks
        @test Y1 == Y2
        # row_bucketed (auto at bs = 1) has no range form: a plain closure, as before
        @test !(build_optimized_mapping(H, true, Val(1), a, b).f!_rescaled isa
                UniPolfed.Optimization.RangeKernelOperator)
    end

    @testset "mapping_kernels lists what applies" begin
        out = sprint(io -> mapping_kernels(io, H))
        @test occursin("universal_row_read", out)
        @test occursin("✓", out)
    end
end
