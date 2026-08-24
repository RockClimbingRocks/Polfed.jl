# hopping kernels: both variants against ground truth.
#
# Ground truth is `mul!` on the independently assembled sparse operator, never a snapshot of what
# the kernels produce. The cases are chosen for what they stress, not for coverage:
#
#   D = 1        no orthogonal neighbours at all, and a single x-line — the geometry where the
#                work decomposition has to split INSIDE a line or run on one thread
#   D = 2 … 6    the fully unrolled stencil
#   D = 7, 8     past MAX_UNROLLED_DIM: the counted-loop stencil, which must agree exactly
#   L₁ ∤ P       a pack that would straddle an x-line — `packed` must narrow W, not read across
#   odd L₁       P is forced to 1; the shuffles degenerate to whole-register rotation
#   periodic     every wrap bond, which is where an off-by-one hides

using LinearAlgebra, SparseArrays
import UniPolfed.Optimization: HopStructure, classify, rescale, build_kernel, build_hop_kernel,
                               MappingKernel, _derive_hop, _hop_lane_width, verify_kernel

_hop_ref(H, X) = permutedims(H * permutedims(X))

"""    _packed_expressible(T, bs) — the block sizes `hopping/packed` can build at all."""
_packed_expressible(::Type{T}, bs) where {T} =
    T <: Real && bs <= _hop_lane_width(T) && rem(_hop_lane_width(T), bs) == 0

@testset "hopping kernels" begin

    @testset "matches mul! — $shape, $T, bs=$bs, $variant" for
            shape in ((12,), (6, 5), (4, 5, 3), (3, 4, 3, 3), (3, 3, 3, 3, 3),
                      (3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3)),
            T in (Float64, Float32, ComplexF64),
            bs in (1, 2, 4),
            variant in (:packed, :rows)

        variant === :packed && !_packed_expressible(T, bs) && continue
        H, _ = anderson_sparse(shape; t = T(-1.0), T = T, diags = rand(T, prod(shape)))
        f! = build_kernel(Val(:hopping), Val(variant), classify(H), Val(bs))
        X = rand(T, bs, size(H, 1)); Y = similar(X)
        f!(Y, X)
        tol = real(T) === Float32 ? 1e-4 : 1e-12
        @test maximum(abs.(Y .- _hop_ref(H, X))) < tol * max(1, maximum(abs, X))
    end

    @testset "past MAX_UNROLLED_DIM the looped stencil equals the unrolled one" begin
        # D = 7 and 8 take the counted-loop path; the answer must not depend on the cut
        for shape in ((3, 3, 3, 3, 3, 3, 3), (3, 3, 3, 3, 3, 3, 3, 3))
            H, _ = anderson_sparse(shape; t = -1.0)
            X = rand(2, size(H, 1)); want = _hop_ref(H, X)
            for variant in (:packed, :rows)
                f! = build_kernel(Val(:hopping), Val(variant), classify(H), Val(2))
                Y = similar(X); f!(Y, X)
                @test maximum(abs.(Y .- want)) < 1e-12
            end
        end
    end

    @testset "x-extents a pack does not divide" begin
        # L₁ = 6 admits P ∈ {1,2}; L₁ = 9 forces P = 1. Both must be exact, not merely close.
        for shape in ((6, 4), (9, 4), (10, 3, 3)), bs in (1, 2, 4)
            H, _ = anderson_sparse(shape; t = -1.0)
            X = rand(bs, size(H, 1)); want = _hop_ref(H, X)
            Y = similar(X)
            build_kernel(Val(:hopping), Val(:packed), classify(H), Val(bs))(Y, X)
            @test maximum(abs.(Y .- want)) < 1e-12
        end
    end

    @testset "a pinned W too wide for the lattice is narrowed, not honoured blindly" begin
        H, _ = anderson_sparse((6, 4); t = -1.0)          # P = 8 would straddle the x-line
        X = rand(1, size(H, 1)); Y = similar(X)
        build_kernel(Val(:hopping), Val(:packed), classify(H), Val(1); W = 8)(Y, X)
        @test maximum(abs.(Y .- _hop_ref(H, X))) < 1e-12
    end

    @testset "rows takes the block sizes and element types packed cannot" begin
        H, _ = anderson_sparse((6, 5); t = -1.0)
        s = classify(H); D = size(H, 1)
        for bs in (3, 5, 7)                                # not divisors of the pack width
            f! = build_kernel(Val(:hopping), Val(:rows), s, Val(bs))
            X = rand(bs, D); Y = similar(X); f!(Y, X)
            @test maximum(abs.(Y .- _hop_ref(H, X))) < 1e-12
        end
        @test_throws ArgumentError build_kernel(Val(:hopping), Val(:packed), s, Val(3))

        Hc, _ = anderson_sparse((6, 5); t = ComplexF64(-1.0), T = ComplexF64,
                                diags = rand(ComplexF64, 30))
        f! = build_kernel(Val(:hopping), Val(:rows), classify(Hc), Val(2))
        X = rand(ComplexF64, 2, 30); Y = similar(X); f!(Y, X)
        @test maximum(abs.(Y .- _hop_ref(Hc, X))) < 1e-12
        @test_throws ArgumentError build_kernel(Val(:hopping), Val(:packed), classify(Hc), Val(2))
    end

    @testset "the parallel work split does not change the answer" begin
        # `tile` re-cuts the units; at D = 1 the cut has to fall INSIDE the single x-line
        for shape in ((64,), (12, 12, 12)), tile in (0, 1, 3, 64)
            H, _ = anderson_sparse(shape; t = -1.0)
            X = rand(2, size(H, 1)); want = _hop_ref(H, X)
            for variant in (:packed, :rows)
                f! = build_kernel(Val(:hopping), Val(variant), classify(H), Val(2); tile = tile)
                Y = similar(X); f!(Y, X)
                @test maximum(abs.(Y .- want)) < 1e-12
            end
        end
    end

    @testset "matrix-free build equals the matrix-derived one" begin
        shape = (8, 5, 3); n = prod(shape); diags = randn(n)
        H, _ = anderson_sparse(shape; t = -1.0, diags = diags)
        hs = HopStructure(diags, shape, -1.0)
        for bs in (1, 4)
            X = rand(bs, n); want = _hop_ref(H, X)
            for variant in (:auto, :packed, :rows)
                Y = Matrix{Float64}(undef, bs, n)
                build_hop_kernel(hs, Val(bs); variant = variant)(Y, X)
                @test maximum(abs.(Y .- want)) < 1e-12
            end
        end
        @test_throws ArgumentError build_hop_kernel(hs, Val(1); variant = :nonsense)
    end

    @testset "rescaled structure computes (H - b)/a" begin
        shape = (8, 5); n = prod(shape); diags = randn(n)
        H, _ = anderson_sparse(shape; t = -1.0, diags = diags)
        a, b = 3.0, 0.5
        for bs in (1, 2)
            f! = build_kernel(Val(:hopping), Val(:packed), rescale(classify(H), a, b), Val(bs))
            @test verify_kernel(f!, H, a, b, Val(bs), :hopping_packed)
            f2! = build_hop_kernel(rescale(HopStructure(diags, shape, -1.0), a, b), Val(bs))
            @test verify_kernel(f2!, H, a, b, Val(bs), :hopping_rows)
        end
    end

    @testset "no per-site allocation" begin
        # The invariant is that nothing in the apply scales with the lattice: allocation must be
        # IDENTICAL on a lattice four times the size. Single-threaded that means exactly zero;
        # with threads it is the `@threads` region's own closure, which is a fixed cost per apply
        # and is what the `nthreads() == 1` early return exists to avoid.
        for (small, large) in (((16, 8), (32, 16)), ((6, 6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6, 12)))
            for (bs, variant) in ((1, :packed), (4, :packed), (4, :rows), (3, :rows))
                got = map((small, large)) do shape
                    n = prod(shape)
                    f! = build_hop_kernel(HopStructure(randn(n), shape, -1.0), Val(bs);
                                          variant = variant)
                    X = rand(bs, n); Y = similar(X)
                    f!(Y, X)                                # compile
                    @allocated(f!(Y, X))
                end
                @test got[1] == got[2]
                Threads.nthreads() == 1 && @test got[1] == 0
            end
        end
    end
end
