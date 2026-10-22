using Test
using Polfed
using LinearAlgebra
using Random

const _FC = Polfed.Numerics.Clenshaw

# Ground truth: S(A)·X through the eigendecomposition, S(λ) = Σ cₖ Pₖ(λ) evaluated by the
# (already tested) scalar Clenshaw path.
function _fc_exact(A, X, family, coef, order)
    λ, V = eigen(Hermitian(Matrix(A)))
    s = _FC.ClenshawExpansion(family, coef, order)
    return V * Diagonal(s.(λ)) * V' * X
end

# Symmetric/Hermitian test operator with spectrum inside (-1, 1).
function _fc_operator(E, n; seed = 1)
    M = randn(MersenneTwister(seed), E, n, n)
    A = (M + M') / 2
    return A ./ (1.1 * opnorm(A))
end

# Column-layout range form: units are output rows, `lo:hi` writes Y[lo:hi, :].
_fc_range(A) = (Y, X, lo, hi) -> (mul!(view(Y, lo:hi, :), view(A, lo:hi, :), X); nothing)

@testset "fused Clenshaw evaluators" begin
    coef = k -> cos(0.37k) / (1 + 0.05k)
    n, m = 24, 3

    @testset "AxpbyClenshaw: exact S(A)·X, $E, $family" for E in (Float64, ComplexF64, Float32),
                                                            family in (:Chebyshev, :Legendre, :Taylor)
        tol = E <: Union{Float32,ComplexF32} ? 1e-4 : 1e-11
        A = _fc_operator(E, n)
        X = randn(MersenneTwister(2), E, n, m)
        axpby!(Y, Z, α, β) = (mul!(Y, A, Z, α, β); nothing)
        add!(Y, c, Z) = (axpy!(c, Z, Y); nothing)
        for order in (0, 1, 2, 17)
            e = _FC.AxpbyClenshaw(family, coef, order, axpby!, add!)
            b = [fill(E(NaN), n, m) for _ in 1:3]     # garbage workspace and output
            Y = e(fill(E(NaN), n, m), X, b)
            @test Y ≈ _fc_exact(A, X, family, coef, order) rtol = tol
        end
    end

    @testset "RangeClenshaw: exact, and bit-identical to ClenshawExpansion" begin
        for E in (Float64, ComplexF64), family in (:Chebyshev, :Taylor), order in (0, 1, 17)
            A = _fc_operator(E, n; seed = 3)
            X = randn(MersenneTwister(4), E, n, m)
            ref = _FC.ClenshawExpansion(family, coef, order, (Y, Z) -> mul!(Y, A, Z))(
                zeros(E, n, m), X, [zeros(E, n, m) for _ in 1:3])
            @test ref ≈ _fc_exact(A, X, family, coef, order) rtol = 1e-11
            for ntasks in (1, 2, 3), alignment in (1, 4)
                e = _FC.RangeClenshaw(family, coef, order, _fc_range(A), n;
                                      alignment = alignment, ntasks = ntasks)
                Y = e(fill(E(NaN), n, m), X, [fill(E(NaN), n, m) for _ in 1:3])
                @test Y == ref
            end
        end
    end

    @testset "chunks tile the range on aligned boundaries" begin
        for n in (1, 7, 64, 1000), align in (1, 4, 16), nt in (1, 3, 8)
            chunks = [_FC._chunk_range(n, align, nt, t) for t in 1:nt]
            covered = reduce(vcat, [collect(lo:hi) for (lo, hi) in chunks])
            @test covered == collect(1:n)
            @test all(((lo, hi),) -> lo > hi || (lo - 1) % align == 0, chunks)
        end
    end

    @testset "argument validation" begin
        @test_throws ArgumentError _FC.RangeClenshaw(:Chebyshev, coef, 3, _fc_range(zeros(2, 2)), 0)
        @test_throws ArgumentError _FC.RangeClenshaw(:Nope, coef, 3, _fc_range(zeros(2, 2)), 2)
        @test_throws ArgumentError _FC.AxpbyClenshaw(:Nope, coef, 3, identity, identity)
        e = _FC.RangeClenshaw(:Chebyshev, coef, 3, _fc_range(zeros(4, 4)), 4)
        @test_throws DimensionMismatch e(zeros(4, 2), zeros(4, 2), [zeros(4, 3) for _ in 1:3])
    end

    @testset "type stability" begin
        A = _fc_operator(Float64, 8)
        X = rand(8, 2); b = [zeros(8, 2) for _ in 1:3]
        e = _FC.RangeClenshaw(:Chebyshev, coef, 5, _fc_range(A), 8)
        @inferred e(zeros(8, 2), X, b)
        f = _FC.AxpbyClenshaw(:Chebyshev, coef, 5, (Y, Z, α, β) -> mul!(Y, A, Z, α, β),
                              (Y, c, Z) -> axpy!(c, Z, Y))
        @inferred f(zeros(8, 2), X, b)
    end

    # The threaded region and its barrier only exist above one thread.
    if Threads.nthreads() > 1
        @testset "an error in one task is raised, not a hang" begin
            A = _fc_operator(Float64, 256)
            bad = (Y, X, lo, hi) -> (lo > 1 ? error("boom in a chunk") : mul!(view(Y, lo:hi, :), view(A, lo:hi, :), X))
            e = _FC.RangeClenshaw(:Chebyshev, coef, 5, bad, 256; ntasks = 2)
            @test_throws Exception e(zeros(256, 2), rand(256, 2), [zeros(256, 2) for _ in 1:3])
        end

        @testset "the spin barrier separates rounds" begin
            nt = min(Threads.nthreads(), 4)
            barrier = _FC._SpinBarrier(nt)
            seen = zeros(Int, nt, 50)
            counter = Threads.Atomic{Int}(0)
            Threads.@threads :static for t in 1:nt
                for r in 1:50
                    Threads.atomic_add!(counter, 1)
                    _FC._sync!(barrier)
                    seen[t, r] = counter[]         # everyone has arrived for round r …
                    _FC._sync!(barrier)            # … and nobody has started round r + 1
                end
            end
            @test seen == [nt * r for t in 1:nt, r in 1:50]
        end
    end
end
