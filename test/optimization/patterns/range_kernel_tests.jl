# The range seam: every kernel that claims a serial range form must produce EXACTLY the same
# `Y` as its whole-block form.
#
# Ground truth here is the whole-block kernel rather than `mul!`, deliberately: the point of a
# range form is that it is the same computation with the threading lifted out, so any difference
# is a bug in the split, not in the arithmetic. `mul!` agreement is already covered per pattern.
#
# What the cases stress:
#   uneven splits    chunk counts that do not divide the unit count, so the last chunk is short
#   alignment        spin_flip/packed steps in whole packs; a chunk starting off a pack boundary
#                    is a correctness error and must THROW, not silently compute a shifted answer
#   full coverage    every unit written exactly once — a dropped tail leaves stale data in `Y`,
#                    which surfaces much later as a wrong eigenvector
#   opt-in           a kernel with no range form says so rather than being given a wrong one

using LinearAlgebra, SparseArrays, Random
import Polfed.Optimization: MappingKernel, classify, build_kernel, build_range_kernel,
                               range_units, range_alignment, supports_range

"""    _chunks(n, a, k) -> Vector{UnitRange} — split `1:n` into ~`k` chunks aligned to `a`."""
function _chunks(n::Int, a::Int, k::Int)
    step = max(a, a * cld(cld(n, k), a))
    out = UnitRange{Int}[]
    lo = 1
    while lo <= n
        hi = min(lo + step - 1, n)
        push!(out, lo:hi)
        lo = hi + 1
    end
    return out
end

@testset "range kernels" begin

    @testset "range form equals whole-block form — $pat/$var, $T, bs=$bs, $nchunk chunks" for
            (pat, var, mk) in (
                (:spin_flip, :packed,   T -> qrem(8; T = T)),
                (:spin_flip, :rows,     T -> qrem(8; T = T)),
                (:universal, :row_read, T -> qrem(8; T = T)),
                (:hopping,   :packed,   T -> first(anderson_sparse((6, 5); t = T(-1.0), T = T))),
                (:hopping,   :rows,     T -> first(anderson_sparse((6, 5); t = T(-1.0), T = T)))),
            T in (Float64, Float32, ComplexF64),
            bs in (1, 2, 4),
            nchunk in (1, 3, 7)

        # packed variants are SIMD-only: real eltype, bs a power of two within the lane width
        (var === :packed && !(T <: Real)) && continue

        H = mk(T)
        s = classify(H)
        k = MappingKernel(pat, var)
        supports_range(k) || continue

        f!  = build_kernel(k, s, Val(bs))
        fr! = build_range_kernel(k, s, Val(bs))
        n   = range_units(k, s, Val(bs))
        a   = range_alignment(k, s, Val(bs))

        D = size(H, 1)
        X = rand(T, bs, D)
        Y_whole = zeros(T, bs, D)
        Y_range = zeros(T, bs, D)
        f!(Y_whole, X)
        for r in _chunks(n, a, nchunk)
            fr!(Y_range, X, first(r), last(r))
        end

        tol = real(T) === Float32 ? 1e-5 : 1e-13
        @test maximum(abs.(Y_range .- Y_whole)) < tol * max(1, maximum(abs, Y_whole))
    end

    @testset "chunks tile the unit range exactly once" begin
        # A range form is only usable for a decomposition if distinct chunks write DISJOINT parts
        # of Y and together cover everything. Checked on the counting level, independent of any
        # kernel: a dropped or doubled unit is the failure mode that survives a norm comparison
        # when the missed entries happen to be small.
        for (n, a, k) in ((256, 64, 3), (256, 1, 7), (30, 4, 4), (17, 1, 5))
            rs = _chunks(n, a, k)
            @test first(first(rs)) == 1
            @test last(last(rs)) == n
            @test all(first(rs[i + 1]) == last(rs[i]) + 1 for i in 1:(length(rs) - 1))
            @test all((first(r) - 1) % a == 0 for r in rs)
        end
    end

    @testset "spin_flip/packed rejects a misaligned chunk start" begin
        # Alignment is a correctness precondition of the lane permutation, not a preference. An
        # unaligned start must throw: silently computing a shifted answer is the worst outcome.
        H = qrem(8)
        s = classify(H)
        k = MappingKernel(:spin_flip, :packed)
        a = range_alignment(k, s, Val(2))
        @test a > 1
        fr! = build_range_kernel(k, s, Val(2))
        X = rand(2, size(H, 1)); Y = zeros(2, size(H, 1))
        @test_throws ArgumentError fr!(Y, X, 2, a)          # lo-1 = 1, not a multiple of a
        @test fr!(Y, X, 1, a) === nothing                   # aligned start is accepted
    end

    @testset "a kernel without a range form says so" begin
        # Opt-in, so an unsplit kernel stays safe rather than silently gaining a wrong range.
        k = MappingKernel(:universal, :row_bucketed)
        @test !supports_range(k)
        s = classify(qrem(6))
        @test_throws ArgumentError build_range_kernel(k, s, Val(1))
        @test_throws ArgumentError range_units(k, s, Val(1))
    end
end
