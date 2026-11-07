# MappingKernel: the user's choice, validated against the registry at CONSTRUCTION — before any
# Hamiltonian is involved. The point is that a typo fails immediately and legibly.

@testset "MappingKernel" begin

    @testset "construction" begin
        k = MappingKernel(:universal, :row_read)
        @test k.pattern === :universal
        @test k.variant === :row_read
        @test kernel_name(k) === :universal_row_read

        # the pattern alone picks its default variant — row_read, the one with no restrictions
        d = MappingKernel(:universal)
        @test d.pattern === :universal && d.variant === :row_read

        b = MappingKernel(:universal, :row_bucketed)
        @test kernel_name(b) === :universal_row_bucketed

        # neither kernel takes tuning parameters; params carries the kernel's declared defaults
        @test k.params == NamedTuple() && b.params == NamedTuple()
    end

    @testset "rejects what it cannot honour" begin
        # a parameter this kernel does not declare — these kernels declare none at all
        @test_throws ArgumentError MappingKernel(:universal, :row_read; W = 8)
        @test_throws ArgumentError MappingKernel(:universal; G = 4)
        # unknown pattern / variant
        @test_throws ArgumentError MappingKernel(:nonexistent)
        @test_throws ArgumentError MappingKernel(:universal, :nonexistent)
    end

    @testset "prints as something you could paste back" begin
        @test occursin("MappingKernel(:universal, :row_read)",
                       sprint(show, MappingKernel(:universal)))
        @test occursin("MappingKernel(:universal, :row_bucketed)",
                       sprint(show, MappingKernel(:universal, :row_bucketed)))
    end

    @testset "registry is machine-readable" begin
        @test all(i isa KernelInfo for i in REGISTRY)
        # a bare pattern name resolves to that pattern's FIRST registered variant
        @test MappingKernel(:universal).variant === :row_read
        @test MappingKernel(:spin_flip).variant === :packed
        variants = [(i.pattern, i.variant) for i in REGISTRY]
        @test (:universal, :row_bucketed) in variants
        @test (:spin_flip, :rows) in variants
    end
end
