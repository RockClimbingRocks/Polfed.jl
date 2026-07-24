using Test
using LinearAlgebra
using Random
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

# Build a factorization and run `k` Lanczos iterations (mirrors lanczos_algorithm's
# driver loop). Returns (factorization, ok) where `ok=false` signals breakdown.
function _run_factorization(A, x0, k; rot=LZ.FullRO(), maxdim=64)
    pu = CPU()
    seed = x0 isa AbstractVector ? x0 ./ norm(x0) : Matrix(qr(x0).Q)
    f!(y, x) = mul!(y, A, x)
    iter = LZ.LanczosIterator(f!, seed, rot)
    basis = LZ.createbasis(maxdim, seed, MatrixBasis, pu)
    fac = LZ.lanczositer(iter, basis, maxdim, pu)

    # No @stages scope open here: the `@stage` annotations inside `lanczositer!` are pass-throughs,
    # so the recurrence runs untimed (timing is opt-in by the caller).
    ok = true
    for it in 2:(k+1)
        ok = LZ.lanczositer!(iter, fac, it)
        ok || break
    end
    return fac, ok
end

@testset "Factorization (recurrence correctness)" begin

    @testset "scalar — Ritz–Galerkin identity T = V'AV with full reorthogonalization" begin
        Random.seed!(1234)
        n = 12
        B = randn(n, n); A = Symmetric(B + B')
        x0 = Float64[1/(i + 0.5) - 0.2cos(1.3i) for i in 1:n]

        fac, ok = _run_factorization(A, x0, 6)
        @test ok                                       # no premature breakdown
        V = Matrix(LZ.allvecs(fac.basis))
        T = Matrix(LZ.constructfactorizedmat(fac))

        @test V' * V ≈ I(size(V, 2)) atol=1e-10        # basis orthonormal
        @test T ≈ V' * Matrix(A) * V atol=1e-10        # projected matrix is the Galerkin projection
        @test issymmetric(round.(T; digits=10))        # tridiagonal symmetric
    end

    @testset "scalar — full Krylov space recovers the exact spectrum" begin
        Random.seed!(55)
        n = 10
        B = randn(n, n); A = Symmetric(B + B')
        x0 = Float64[1/(i + 0.3) + 0.1i for i in 1:n]

        # n-1 iterations → krylovdim = n → full space.
        fac, ok = _run_factorization(A, x0, n - 1)
        @test ok
        @test fac.krylovdim == n
        T = Matrix(LZ.constructfactorizedmat(fac))
        @test sort(eigvals(T)) ≈ sort(eigvals(Matrix(A))) atol=1e-10
    end

    @testset "block — Ritz–Galerkin identity and exact spectrum at full space" begin
        Random.seed!(321)
        n = 8; s = 2
        B = randn(n, n); A = Symmetric(B + B')
        x0 = randn(n, s)

        # k=3 block iterations → krylovdim = (k+1)·s = 8 = n → full space.
        fac, ok = _run_factorization(A, x0, 3)
        @test ok
        @test fac.krylovdim == n
        V = Matrix(LZ.allvecs(fac.basis))
        T = Matrix(LZ.constructfactorizedmat(fac))

        @test V' * V ≈ I(n) atol=1e-10
        @test T ≈ V' * Matrix(A) * V atol=1e-10
        @test sort(eigvals(Hermitian(T))) ≈ sort(eigvals(Matrix(A))) atol=1e-10
    end

    @testset "scalar — breakdown when the seed is an exact eigenvector" begin
        n = 6
        λ = Float64[1, 2, 3, 4, 5, 6]
        Random.seed!(11)
        Q = Matrix(qr(randn(n, n)).Q)
        A = Symmetric(Q * Diagonal(λ) * Q')
        evec = Q[:, 3]                                 # exact eigenvector (λ = 3)

        # First lanczositer! must detect breakdown: r = A·v - α·v = 0.
        fac, ok = _run_factorization(A, evec, 1)
        @test ok == false
        @test fac.krylovdim == 1                       # no new vector was appended
    end

    @testset "block — breakdown when the seed block is an invariant subspace" begin
        n = 6
        λ = Float64[1, 2, 3, 4, 5, 6]
        Random.seed!(13)
        Q = Matrix(qr(randn(n, n)).Q)
        A = Symmetric(Q * Diagonal(λ) * Q')
        block = Q[:, 1:2]                              # invariant 2-dim eigenspace

        fac, ok = _run_factorization(A, block, 1)
        @test ok == false
        @test fac.krylovdim == 2                       # only the seed block present
    end

    @testset "getnorm breakdown guard returns zero coupling" begin
        # Right after init (krylovdim == blocksize) getnorm has no coupling to report.
        n = 5
        A = Symmetric([2.0*(i==j) + 0.3*(abs(i-j)==1) for i in 1:n, j in 1:n])
        x0 = normalize(Float64[1, -1, 1, -1, 1])
        fac, _ = _run_factorization(A, x0, 0)          # only initialization
        @test fac.krylovdim == 1
        @test LZ.getnorm(fac) == 0.0                   # scalar: no β yet
    end
end
