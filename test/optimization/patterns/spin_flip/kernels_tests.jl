# spin_flip kernels: both variants against ground truth.
#
# Ground truth is `mul!` on the sparse operator, not a snapshot, and BOTH model shapes are
# covered because they stress different things:
#   sx  — single-bit masks, one shared value (QREM/TFIM)
#   sun — multi-bit masks, all values different (Quantum Sun). This is the case that breaks the
#         value-grouped assumption, and the one where the payoff is largest.

using LinearAlgebra, SparseArrays
import Polfed.Optimization: FlipStructure, classify, rescale, build_kernel, MappingKernel,
                               build_flip_kernel, _derive_flip, _packed_lane_width

_flip_ref(H, X) = permutedims(H * permutedims(X))

@testset "spin_flip kernels" begin

    @testset "matches mul! — $name, $T, bs=$bs, variant=$variant" for
            (name, Hf) in ("sx"  => (T -> qrem(8; T = T)),
                           "sun" => (T -> sun_sparse(8; T = T)[1])),
            T in (Float64, Float32),
            bs in (1, 2, 4, 8),
            variant in (:packed, :rows)

        H = Hf(T)
        s = classify(H)
        f! = build_kernel(Val(:spin_flip), Val(variant), s, Val(bs))
        X = rand(T, bs, size(H, 1)); Y = similar(X)
        f!(Y, X)
        tol = T === Float32 ? 1e-4 : 1e-12
        @test maximum(abs.(Y .- _flip_ref(H, X))) < tol * max(1, maximum(abs, X))
    end

    @testset "rows takes the block sizes packed cannot" begin
        H = qrem(8); s = classify(H); D = size(H, 1)
        for bs in (3, 5, 7)                       # not powers of two
            f! = build_kernel(Val(:spin_flip), Val(:rows), s, Val(bs))
            X = rand(bs, D); Y = similar(X)
            f!(Y, X)
            @test maximum(abs.(Y .- _flip_ref(H, X))) < 1e-12
        end
        # packed refuses them rather than computing a wrong permutation
        @test_throws ArgumentError build_kernel(Val(:spin_flip), Val(:packed), s, Val(3))
    end

    @testset "rows handles complex, which packed cannot express" begin
        H = qrem(7; T = ComplexF64); s = classify(H); D = size(H, 1)
        f! = build_kernel(Val(:spin_flip), Val(:rows), s, Val(2))
        X = rand(ComplexF64, 2, D); Y = similar(X)
        f!(Y, X)
        @test maximum(abs.(Y .- _flip_ref(H, X))) < 1e-12
    end

    @testset "small L, where one G-block would overrun the space" begin
        # P·G can exceed D at the tiny sizes tests use; the kernel must shrink G, not read past
        W = _packed_lane_width(Float64)
        for L in (4, 5, 6)
            D = 1 << L
            D < W * 4 || continue                 # only interesting while P·G > D
            H = qrem(L); s = classify(H)
            f! = build_kernel(Val(:spin_flip), Val(:packed), s, Val(1))
            X = rand(1, D); Y = similar(X)
            f!(Y, X)
            @test maximum(abs.(Y .- _flip_ref(H, X))) < 1e-12
        end
    end

    @testset "the two variants agree with each other exactly" begin
        # Same terms in the same order per row, so this is bit-for-bit, not just within tolerance.
        H = sun_sparse(9)[1]; s = classify(H); D = size(H, 1)
        for bs in (1, 2, 4)
            X = rand(bs, D); Yp = similar(X); Yr = similar(X)
            build_kernel(Val(:spin_flip), Val(:packed), s, Val(bs))(Yp, X)
            build_kernel(Val(:spin_flip), Val(:rows), s, Val(bs))(Yr, X)
            @test Yp == Yr
        end
    end

    @testset "kernels are allocation-free once warm" begin
        # Single-threaded only. Above one thread the kernels take their `Threads.@threads :static`
        # branch, and spawning that region costs ~450 B per thread (3840 B at nt = 8) no matter
        # what the kernel body does. The claim being guarded is that the KERNEL allocates nothing,
        # which is only observable on the serial path.
        if Threads.nthreads() > 1
            @test_skip "allocation-free check is single-threaded only"
        else
            H = qrem(9); s = classify(H); D = size(H, 1)
            X = rand(1, D); Y = similar(X)
            for variant in (:packed, :rows)
                f! = build_kernel(Val(:spin_flip), Val(variant), s, Val(1))
                f!(Y, X)                                        # warm up
                @test (@allocated f!(Y, X)) == 0
            end
        end
    end

    @testset "W and G are tunable and never change the answer" begin
        # They trade vector width against register pressure, so the optimum is machine- and
        # bs-dependent; what must NOT depend on them is the result.
        H = sun_sparse(9)[1]; s = classify(H); D = size(H, 1)
        for bs in (1, 2, 4)
            X = rand(bs, D); ref = _flip_ref(H, X); Y = similar(X)
            for W in (0, 8, 16), G in (1, 2, 4, 8)
                W == 0 || W >= bs || continue
                f! = build_kernel(MappingKernel(:spin_flip, :packed; W = W, G = G), s, Val(bs))
                f!(Y, X)
                @test maximum(abs.(Y .- ref)) < 1e-12
            end
            for G in (1, 2, 4, 8, 16)
                f! = build_kernel(MappingKernel(:spin_flip, :rows; G = G), s, Val(bs))
                f!(Y, X)
                @test maximum(abs.(Y .- ref)) < 1e-12
            end
        end
        # the registry knows which kernel accepts which knob
        @test_throws ArgumentError MappingKernel(:spin_flip, :rows; W = 8)
        @test MappingKernel(:spin_flip, :packed).params == (W = 0, G = 4)
        @test MappingKernel(:spin_flip, :packed; G = 2).params == (W = 0, G = 2)
        # nonsense values are refused at build time, not silently accepted
        @test_throws ArgumentError build_kernel(MappingKernel(:spin_flip, :packed; W = 3),
                                                s, Val(1))
        @test_throws ArgumentError build_kernel(MappingKernel(:spin_flip, :rows; G = 0),
                                                s, Val(1))
    end

    @testset "matrix-free construction gives the same operator" begin
        # The whole point: no sparse H is ever assembled.
        L = 9; D = 1 << L
        diags = randn(D)
        masks = [1 | (1 << j) for j in 1:L-1]
        values = [0.75^(j - 1) for j in 1:L-1]
        fs = FlipStructure(diags, masks, values)

        rowi = Int[]; colj = Int[]; vals = Float64[]     # not `I` — that shadows the identity
        for p in 0:D-1
            push!(rowi, p+1); push!(colj, p+1); push!(vals, diags[p+1])
            for (m, v) in zip(masks, values)
                push!(rowi, p+1); push!(colj, xor(p, m) + 1); push!(vals, v)
            end
        end
        H = sparse(rowi, colj, vals, D, D)

        for bs in (1, 4), variant in (:auto, :packed, :rows)
            f! = build_flip_kernel(fs, Val(bs); variant = variant)
            X = rand(bs, D); Y = similar(X)
            f!(Y, X)
            @test maximum(abs.(Y .- _flip_ref(H, X))) < 1e-12
        end

        # and it rescales like the matrix path does
        a, b = 3.0, 0.5
        f! = build_flip_kernel(rescale(fs, a, b), Val(1))
        X = rand(1, D); Y = similar(X)
        f!(Y, X)
        @test maximum(abs.(Y .- _flip_ref((H - b * I) / a, X))) < 1e-12

        @test_throws ArgumentError build_flip_kernel(fs, Val(1); variant = :nope)
    end
end
