# End-to-end tests for the WINDOW-target entrypoints `polfed(H|f!, x0, target)` — the mode that
# takes an energy *window* (no `howmany`) and returns ALL eigenpairs whose true energy lands inside
# it. Ground truth is exact diagonalization: for an interior window whose bounds fall in spectral
# GAPS (so the enclosed set is unambiguous — the bounds are not themselves eigenvalues, sidestepping
# the closed-interval boundary-rounding drop), POLFED must return exactly the true eigenvalues of H
# in [lo, hi], to machine precision. This formalizes the count→extrema→window roundtrip validated by
# the `window_roundtrip_QREM` notebook.

using LinearAlgebra, SparseArrays, Random, Logging, Test
using UniPolfed: polfed, MappingConfig, PolfedResult

# QREM: diag(ε) + hx·Σᵢ σᵢˣ (see matrix_entrypoint_tests / qrem_fixture). Real-symmetric.
function _qrem_w(L::Int; hx = 0.7, T = Float64)
    D = 2^L
    diagv = randn(T, D)
    I = Int[]; J = Int[]; V = T[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, T(hx))
        end
    end
    return sparse(I, J, V, D, D)
end

# `runwin` runs a window-mode call while CAPTURING its logs, then asserts the completeness certificate
# fired correctly: for these fully-captured interior windows the "may be INCOMPLETE" warning must NOT
# appear — the converged run straddles both edges, so the criterion certifies the interval and
# reports it via `state.complete` / `FactorizationReport.complete`. This replaces blanket log
# suppression so the test actively guards against a false-positive incompleteness warning.
function runwin(f)
    logger = Test.TestLogger()
    res = with_logger(f, logger)
    @test !any(r -> occursin("INCOMPLETE", r.message), logger.logs)
    return res
end

@testset "window entrypoint polfed(…, target::window)" begin
    Random.seed!(20260714)
    L = 8; D = 2^L
    H = _qrem_w(L)
    exact = sort(eigvals(Symmetric(Matrix(H))))

    # Interior block of the spectrum; window bounds placed in the GAPS on either side so the closed
    # interval [lo, hi] unambiguously encloses exactly exact[i0:i1]. The bounds are midpoints of the
    # bracketing gaps — not eigenvalues — so ~1e-12 reconstruction rounding cannot move the count.
    i0, i1 = 100, 119                         # 20 interior eigenvalues
    lo = (exact[i0-1] + exact[i0]) / 2
    hi = (exact[i1]   + exact[i1+1]) / 2
    want = exact[i0:i1]

    # POLFED must return exactly the true eigenvalues of H inside [lo, hi], ascending — the count
    # equality is itself the completeness proof (see the note above on the conservative warning) —
    # plus one BRACKET eigenpair on each side, so the interval comes back visibly enclosed.
    #
    # The brackets are asserted only to exist and to lie on the correct side, NOT to be the
    # immediate spectral neighbours. A bracket sits just outside the passband, where the Chebyshev
    # filter's tail oscillates: P is monotone in |E − centre| INSIDE the passband (which is what
    # makes the in-window set provably complete) but not below it, so the converged run can reach a
    # distant state before a nearer one. Measured on this fixture: E = −0.9065 has P = 0.127 while
    # the true neighbour E = −0.5516 has P = 0.112.
    function check(res)
        @test res isa PolfedResult
        vals = sort(collect(res.vals))
        inwin = filter(x -> lo <= x <= hi, vals)
        @test length(inwin) == length(want)
        @test length(inwin) == length(want) && maximum(abs.(inwin .- want)) < 1e-8
        @test count(<(lo), vals) == 1        # bracketed below
        @test count(>(hi), vals) == 1        # bracketed above
    end

    @testset "matrix entrypoint, unrescaled window, optimize=$(repr(choice))" for choice in
            (false, :ell32_svec, :shuf)
        res = runwin() do
            polfed(H, randn(D), (:unrescaled, (lo, hi)); mc = MappingConfig(optimize_mapping = choice))
        end
        check(res)
    end

    @testset "bare (lo, hi) tuple defaults to unrescaled" begin
        res = runwin(() -> polfed(H, randn(D), (lo, hi)))
        check(res)
    end

    @testset "operator (column-native) entrypoint matches" begin
        f! = (Y, X) -> mul!(Y, H, X)
        res = runwin(() -> polfed(f!, randn(D), (:unrescaled, (lo, hi))))
        check(res)
    end

    @testset "block seed (bs=2) window" begin
        res = runwin() do
            polfed(H, randn(D, 2), (:unrescaled, (lo, hi)); mc = MappingConfig(optimize_mapping = :shuf))
        end
        check(res)
    end

    @testset "point target is rejected by the window entrypoint" begin
        @test_throws Exception polfed(H, randn(D), :maxdos)
    end
end
