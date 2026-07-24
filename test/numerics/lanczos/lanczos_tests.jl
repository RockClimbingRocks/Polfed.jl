using Test
using UniPolfed
using UniPolfed.Numerics.Lanczos
using UniPolfed.Common: verbosity, POLFED_WARN_LEVEL
using LinearAlgebra
using Logging
using Random

# Reproducible test matrix; deterministic generic seeds (never an eigenvector of A,
# so Lanczos never degenerately breaks down at iteration 1).
Random.seed!(20240601)

function make_sym(λ)
    n = length(λ)
    Q = Matrix(qr(randn(n, n)).Q)
    A = Q * Diagonal(λ) * Q'
    return Matrix(Hermitian((A + A') / 2))
end

genseed(::Type{T}, n) where {T<:Real}    = T[1 / (i + 0.5) - 0.3cos(1.3i) + 0.2 for i in 1:n]
genseed(::Type{T}, n) where {T<:Complex} = T[(1 / (i + 0.5) + 0.2) + 0.3cos(1.3i) * im for i in 1:n]
genblock(::Type{T}, n) where {T} = T[1 / (i + j + 0.5) - 0.2cos(1.3i + 0.7j) for i in 1:n, j in 1:2]

@testset "lanczos" begin
    λ = sort(vcat([-10.0, -8.0, -6.0, 6.0, 8.0, 10.0], range(-1.0, 1.0; length=34)))
    n = length(λ)                       # 40
    A = make_sym(λ)
    exact = eigen(A).values             # ascending
    x0 = genseed(Float64, n)
    md = n                              # full budget → guaranteed convergence on this size

    @testset "LowestEigs / HighestEigs" begin
        r = lanczos(A, x0, LowestEigs(3); maxdim=md)
        @test r.converged == 3
        @test sort(r.vals) ≈ exact[1:3]

        r = lanczos(A, x0, HighestEigs(3); maxdim=md)
        @test sort(r.vals) ≈ exact[end-2:end]
    end

    @testset "BothEnds — count (one factorization, both extremes)" begin
        r = lanczos(A, x0, BothEnds(2, 2); maxdim=md)
        @test r.converged == 4
        @test sort(r.vals) ≈ vcat(exact[1:2], exact[end-1:end])
    end

    @testset "Lowest / BothEnds — ToValue (all eigenpairs out to a value)" begin
        r = lanczos(A, x0, Lowest(ToValue(-5.0)); maxdim=md)
        @test sort(r.vals) ≈ [-10.0, -8.0, -6.0]

        r = lanczos(A, x0, BothEnds(ToValue(-5.0), ToValue(5.0)); maxdim=md)
        @test sort(r.vals) ≈ [-10.0, -8.0, -6.0, 6.0, 8.0, 10.0]
    end

    @testset "warn_incomplete gates the budget-exhausted warning" begin
        # A Krylov budget far too small to certify 20 eigenpairs finalizes at maxdim without
        # `done` — the condition that emits the "budget exhausted" warning.
        saved = verbosity[]
        try
            verbosity[] = POLFED_WARN_LEVEL
            @test_logs (:warn,) match_mode = :any lanczos(A, x0, LowestEigs(20); maxdim = 6)
            # Same run, warning opted out: no warning, and the (partial) result is unchanged.
            r_off = @test_logs min_level = Logging.Warn lanczos(A, x0, LowestEigs(20);
                                                                maxdim = 6, warn_incomplete = false)
            r_on  = lanczos(A, x0, LowestEigs(20); maxdim = 6)   # suppression is warning-only…
            @test sort(r_off.vals) == sort(r_on.vals)            # …it does not change what converges
        finally
            verbosity[] = saved
        end
    end

    @testset "eigenvectors satisfy A·v ≈ λ·v" begin
        r = lanczos(A, x0, LowestEigs(3); maxdim=md)
        for i in 1:r.converged
            v = r.vecs[:, i]
            @test norm(A * v - r.vals[i] * v) < 1e-7
            @test r.residuals[i] < 1e-7
        end
    end

    @testset "results are returned ascending, vecs/residuals stay aligned" begin
        # BothEnds collects both extremes, so ascending sorting is non-trivial here.
        r = lanczos(A, x0, BothEnds(2, 2); maxdim=md)
        @test issorted(r.vals)
        for i in 1:r.converged
            v = r.vecs[:, i]
            @test norm(A * v - r.vals[i] * v) < 1e-7   # vec[i] matches val[i]
            @test r.residuals[i] < 1e-7                # residual[i] matches pair i
        end
    end

    @testset "_prepare_seed" begin
        @testset "vector is normalized, direction preserved, input untouched" begin
            x = [3.0, 4.0, 0.0]
            s = UniPolfed.Numerics.Lanczos._prepare_seed(UniPolfed.Numerics.Lanczos.ColumnLayout(), x)
            @test norm(s) ≈ 1
            @test s ≈ x / norm(x)
            @test x == [3.0, 4.0, 0.0]
        end

        @testset "single-column matrix is normalized (shape kept)" begin
            x = reshape([3.0, 4.0], 2, 1)
            s = UniPolfed.Numerics.Lanczos._prepare_seed(UniPolfed.Numerics.Lanczos.ColumnLayout(), x)
            @test size(s) == (2, 1)
            @test norm(s) ≈ 1
        end

        @testset "multi-column block is orthonormalized, spans the same space" begin
            Random.seed!(5)
            x = randn(6, 3)
            s = UniPolfed.Numerics.Lanczos._prepare_seed(UniPolfed.Numerics.Lanczos.ColumnLayout(), x)
            @test size(s) == (6, 3)
            @test s' * s ≈ I(3) atol=1e-12
            @test rank(hcat(s, x); atol=1e-10) == 3      # no new directions added
        end

        @testset "complex block is orthonormalized" begin
            Random.seed!(6)
            x = randn(ComplexF64, 5, 2)
            s = UniPolfed.Numerics.Lanczos._prepare_seed(UniPolfed.Numerics.Lanczos.ColumnLayout(), x)
            @test s' * s ≈ I(2) atol=1e-12
        end
    end

    @testset "element types ($T)" for T in (Float64, ComplexF64)
        At = Matrix(Hermitian(T.(A)))
        r = lanczos(At, genseed(T, n), LowestEigs(2); maxdim=md)
        @test sort(real.(r.vals)) ≈ exact[1:2]
        @test eltype(r.vals) == real(T)
    end

    @testset "Float32 (looser tolerance)" begin
        A32 = Matrix(Hermitian(Float32.(A)))
        r = lanczos(A32, genseed(Float32, n), LowestEigs(2); maxdim=md, tol=1e-6, eigentol=1e-4)
        @test isapprox(sort(r.vals), Float32.(exact[1:2]); rtol=1e-3, atol=1e-3)
    end

    @testset "block Lanczos (auto-orthonormalized seed)" begin
        r = lanczos(A, genblock(Float64, n), LowestEigs(2); maxdim=md)
        @test sort(r.vals) ≈ exact[1:2]
    end

    @testset "VectorBasis backend (scalar + block)" begin
        r = lanczos(A, x0, LowestEigs(3); basistype=VectorBasis, maxdim=md)
        @test r.converged == 3
        @test sort(r.vals) ≈ exact[1:3]

        rb = lanczos(A, genblock(Float64, n), LowestEigs(2); basistype=VectorBasis, maxdim=md)
        @test sort(rb.vals) ≈ exact[1:2]
    end

    @testset "VectorBasis vs MatrixBasis agree ($T)" for T in (Float64, ComplexF64, Float32)
        tol  = T <: Union{Float32,ComplexF32} ? 1e-6 : 1e-14
        etol = T <: Union{Float32,ComplexF32} ? 1e-4 : 1e-10
        rtol = T <: Union{Float32,ComplexF32} ? 1e-4 : 1e-10
        At = Matrix(Hermitian(T.(A)))

        # scalar
        rm = lanczos(At, genseed(T, n), LowestEigs(3); basistype=MatrixBasis, maxdim=md, tol=tol, eigentol=etol)
        rv = lanczos(At, genseed(T, n), LowestEigs(3); basistype=VectorBasis, maxdim=md, tol=tol, eigentol=etol)
        @test eltype(rv.vals) == real(T)
        @test sort(real.(rv.vals)) ≈ sort(real.(rm.vals)) rtol=rtol
        @test sort(real.(rv.vals)) ≈ real.(exact[1:3]) rtol=rtol

        # block
        rmb = lanczos(At, genblock(T, n), LowestEigs(2); basistype=MatrixBasis, maxdim=md, tol=tol, eigentol=etol)
        rvb = lanczos(At, genblock(T, n), LowestEigs(2); basistype=VectorBasis, maxdim=md, tol=tol, eigentol=etol)
        @test sort(real.(rvb.vals)) ≈ sort(real.(rmb.vals)) rtol=rtol
    end

    @testset "Int convenience form (which)" begin
        @test sort(lanczos(A, x0, 3; which=:SR, maxdim=md).vals) ≈ exact[1:3]
        @test sort(lanczos(A, x0, 3; which=:LR, maxdim=md).vals) ≈ exact[end-2:end]
        @test_throws ArgumentError lanczos(A, x0, 3; which=:XX, maxdim=md)
    end

    @testset "operator (f!) interface + backward-compatible destructuring" begin
        f!(y, x) = mul!(y, A, x)
        r = lanczos(f!, x0, LowestEigs(2); maxdim=md)
        @test sort(r.vals) ≈ exact[1:2]
        vals, vecs, report = r          # LanczosResult iterates as a tuple
        @test length(vals) == r.converged
        @test size(vecs, 2) == r.converged
        @test report isa UniPolfed.Numerics.Lanczos.FactorizationReport
    end

    @testset "lanczos_extrema" begin
        Emin, Emax = lanczos_extrema(A; x0=x0, maxdim=200)
        @test Emin ≈ exact[1]
        @test Emax ≈ exact[end]

        # modest default maxdim still nails both extremes (BothEnds, one factorization)
        Emin2, Emax2 = lanczos_extrema(A; x0=x0)
        @test Emin2 ≈ exact[1]
        @test Emax2 ≈ exact[end]

        # power_iters = 0 disables the warm-up; still correct
        Emin3, Emax3 = lanczos_extrema(A; x0=x0, power_iters=0)
        @test Emin3 ≈ exact[1]
        @test Emax3 ≈ exact[end]

        # operator (f!) interface
        f!(Y, X) = mul!(Y, A, X)
        Emin4, Emax4 = lanczos_extrema(f!, x0)
        @test Emin4 ≈ exact[1]
        @test Emax4 ≈ exact[end]

        # return_result=true yields the full LanczosResult (both extremes converged)
        res = lanczos_extrema(A; x0=x0, return_result=true)
        @test res isa LanczosResult
        @test res.converged == 2
    end

    @testset "invariant-subspace seed (breakdown is handled)" begin
        # seeding with an exact eigenvector makes the Krylov space 1-dimensional →
        # Lanczos breaks down at iteration 1; the eigenpair must still be returned.
        evec = eigen(A).vectors[:, 1]
        r = lanczos(A, evec, LowestEigs(2); maxdim=md)
        @test r.converged ≥ 1
        @test minimum(abs.(r.vals .- exact[1])) < 1e-8
    end

    @testset "requested count exceeding the dimension errors (upfront check)" begin
        @test_throws ArgumentError lanczos(A, x0, LowestEigs(n + 1); maxdim=md)
        @test_throws ArgumentError lanczos(A, x0, BothEnds(n, 1); maxdim=md)   # sum n+1 > n
    end

    @testset "iterates until EIGEN-converged, not just Ritz-converged (rule 2)" begin
        # Loose Ritz tol but tight eigen tol: if the loop stopped the moment the Ritz
        # residuals were satisfied, the returned eigenvectors would only be ~1e-4
        # accurate. Requiring the tight eigentol forces extra factorization iterations.
        r = lanczos(A, x0, LowestEigs(3); maxdim=md, tol=1e-4, eigentol=1e-12)
        @test r.converged == 3
        @test all(r.residuals .< 1e-12)             # every returned pair meets the tight gate
    end

    @testset "only fully-converged pairs returned; holes truncate the run (rules 1 & 3)" begin
        # Iterate on the DIAGONAL operator F (eigenvectors = standard basis e_i, the
        # isolated low eigenvalues converge fast), but evaluate eigenpairs against
        # H = F with an off-diagonal coupling on rows 3–4. So e_3 passes the Ritz gate
        # but FAILS the eigen gate (it is not an eigenvector of H), while e_1, e_2 are
        # exact eigenpairs of H. maxdim < n keeps extraction out of the full-space regime.
        F = diagm(λ)                              # λ has isolated lows: -10, -8, -6, ...
        H = copy(F); H[3, 4] = H[4, 3] = 0.5
        f!(y, v) = mul!(y, F, v)
        Hmap!(y, v) = mul!(y, H, v)

        r = lanczos(f!, x0, LowestEigs(5); mapvals=Hmap!, maxdim=30, tol=1e-10, eigentol=1e-8)

        # Rule 1: nothing that fails the eigen gate is handed back.
        @test all(r.residuals .< 1e-8)
        # Rule 3: the hole at e_3 truncates the run to the contiguous prefix e_1, e_2 —
        #         later eigenpairs (e.g. e_5) sit *behind* the hole and are dropped.
        @test r.converged == 2
        @test length(r.vals) == 2
        @test sort(r.vals) ≈ [-10.0, -8.0] atol=1e-7
    end
    @testset "Window target (mapped-value interval, ground truth)" begin
        # A diagonal "transformed" operator A = diag(p) maps a mapped-value window to one extreme of
        # A; mapvals = diag(m) recovers the mapped values. A `Window` target must return the m-values
        # inside [lo, hi] PLUS one bracket on each side — gap-free, both tolerances met.
        Random.seed!(42)
        n = 60
        m  = collect(range(-1.0, 1.0, length = n))          # mapped values (ascending, distinct)
        M!(y, x) = (y .= m .* x)                            # mapvals: recovers the mapped value
        x0 = randn(n)
        c = 0.1701                                          # filter centre, off the grid's midpoint
        lo, hi = c - 0.2, c + 0.2
        # The interval comes back ENCLOSED: everything inside, plus the nearest neighbour on
        # each side, so the caller can see nothing was missed at the edges.
        inwin = m[lo .<= m .<= hi]
        want  = sort(vcat(maximum(m[m .< lo]), inwin, minimum(m[m .> hi])))

        # A sharply-peaked (Lorentzian) filter, as a real Chebyshev filter is. A parabolic
        # peak would be FLAT at the top, clustering the extremal transformed values so
        # tightly that scalar Lanczos cannot separate them — the Ritz residual then plateaus
        # around 1e-3 and no pair ever passes the Ritz gate. Peak centre is off the grid's
        # symmetry point so the ± mirror pairs are not degenerate.
        P(x) = 1 / (1 + ((x - c) / 0.05)^2)

        @testset "Highest end (window → high end)" begin
            p = P.(m)
            A!(y, x) = (y .= p .* x)
            tgt = Highest(Window(lo, hi))
            res = lanczos(A!, x0, tgt; mapvals = M!, maxdim = n, tol = 1e-8, eigentol = 1e-9)
            @test sort(collect(res.vals)) ≈ want atol = 1e-6
            @test maximum(res.residuals) < 1e-8            # returned pairs genuinely eigen-converged
        end

        @testset "Lowest end (filter negated: window → low end)" begin
            p = -P.(m)
            A!(y, x) = (y .= p .* x)
            tgt = Lowest(Window(lo, hi))
            res = lanczos(A!, x0, tgt; mapvals = M!, maxdim = n, tol = 1e-8, eigentol = 1e-9)
            @test sort(collect(res.vals)) ≈ want atol = 1e-6
            @test maximum(res.residuals) < 1e-8
        end
    end

    # FactorizationReport contents/display are tested in FactorizationReport_tests.jl.
end
