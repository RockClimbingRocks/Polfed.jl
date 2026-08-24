# End-to-end tests for the TWO-WINDOW entrypoints `double_polfed(H|f!, x0, [howmany,] t1, t2)`.
#
# Ground truth is exact diagonalization: for two disjoint interior windows whose bounds fall in
# spectral GAPS, both paths — one joint `DoubleDeltaFilter` run and two separate single-filter runs
# — must return exactly the true eigenvalues of H inside each window. Forcing `strategy` is what
# makes the two paths independently testable on the SAME problem; the `:auto` case then only has to
# assert that it picked one of them and got the same answer.

using LinearAlgebra, SparseArrays, Random, Test
using UniPolfed: double_polfed, polfed, MappingConfig, PolfedResult, DoublePolfedResult

# QREM: diag(ε) + hx·Σᵢ σᵢˣ — real-symmetric, no spectral symmetry to trip the filters.
function _qrem_d(L::Int; hx = 0.7, T = Float64)
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

@testset "double_polfed entrypoints — L = $L" for L in (8, 10)
    Random.seed!(20260727)
    D = 2^L
    H = _qrem_d(L)
    exact = sort(eigvals(Symmetric(Matrix(H))))

    # Two interior blocks, each bounded by gap midpoints so the enclosed set is unambiguous.
    gapmid(i) = (exact[i] + exact[i + 1]) / 2
    a0, b0 = cld(D, 4), cld(3D, 4)         # two disjoint interior blocks, 20 states each
    lo1, hi1 = gapmid(a0 - 1), gapmid(a0 + 19)
    lo2, hi2 = gapmid(b0 - 1), gapmid(b0 + 19)
    want1, want2 = exact[a0:a0+19], exact[b0:b0+19]
    w1, w2 = (lo1, hi1), (lo2, hi2)

    # The eigenvalues inside a window must match ED exactly; anything outside it is a bracket and
    # must stay on ITS OWN side of the valley — never the neighbouring window's eigenpair.
    function check_window(res::PolfedResult, want, lo, hi, other_lo, other_hi)
        vals = sort(collect(res.vals))
        inwin = filter(x -> lo <= x <= hi, vals)
        @test length(inwin) == length(want)
        @test length(inwin) == length(want) && maximum(abs.(inwin .- want)) < 1e-8
        @test !any(x -> other_lo <= x <= other_hi, vals)     # no leakage from the other window
    end

    function check_pair(res)
        @test res isa DoublePolfedResult
        @test length(res) == 2
        r1, r2 = res                                          # destructuring must work
        @test r1 === res[1] && r2 === res[2]
        check_window(r1, want1, lo1, hi1, lo2, hi2)
        check_window(r2, want2, lo2, hi2, lo1, hi1)
    end

    @testset "window mode, forced strategy=$(repr(strategy))" for strategy in (:double, :separate)
        res = double_polfed(H, randn(D), w1, w2; strategy = strategy)
        check_pair(res)
        @test res.strategy === strategy
        @test res.reason === :forced
        @test res.order1 > 0 && res.order2 > 0                # both are always resolved…
        @test (res.order_double > 0) == (strategy === :double)  # …K_d only on the joint path
    end

    @testset "window mode, :auto decides and is correct either way" begin
        res = double_polfed(H, randn(D), w1, w2)
        check_pair(res)
        @test res.strategy in (:double, :separate)
        @test res.reason === :ratio
        # The decision must agree with the rule it claims to follow.
        @test (res.strategy === :double) ==
              (res.ratio >= UniPolfed.Defaults.double_polfed.double_ratio)
    end

    @testset "results come back in the CALLER's argument order" begin
        forward  = double_polfed(H, randn(D), w1, w2; strategy = :double)
        reversed = double_polfed(H, randn(D), w2, w1; strategy = :double)
        check_window(reversed[1], want2, lo2, hi2, lo1, hi1)   # first result ↔ first target
        check_window(reversed[2], want1, lo1, hi1, lo2, hi2)
        @test length(forward[1].vals) == length(reversed[2].vals)
    end

    @testset "operator (column-native) entrypoint matches the matrix path" begin
        f! = (Y, X) -> mul!(Y, H, X)
        check_pair(double_polfed(f!, randn(D), w1, w2; strategy = :double))
    end

    @testset "count mode returns the requested eigenpairs at both targets" begin
        # A count target is grown from the DoS into a window around the center, so the returned
        # count can fall a little short of the request and the set can differ from the strict
        # `howmany` nearest by an eigenvalue or two at the edges (the peak sits at the resolved
        # window's center, not exactly on `c` — the same DoS-trim as single `polfed`). What must
        # hold: everything returned is a TRUE eigenvalue, and all of them sit in the immediate
        # neighbourhood of the requested center.
        c1, c2 = exact[70], exact[170]
        for strategy in (:double, :separate)
            res = double_polfed(H, randn(D), 10, c1, c2; strategy = strategy)
            @test res.strategy === strategy
            for (r, c) in ((res[1], c1), (res[2], c2))
                @test length(r.vals) > 0
                @test all(λ -> minimum(abs, exact .- λ) < 1e-8, r.vals)   # true eigenvalues
                neighbourhood = sort(exact, by = λ -> abs(λ - c))[1:25]   # 10 requested, 2.5× slack
                @test all(λ -> any(≈(λ), neighbourhood), r.vals)
            end
        end
    end

    @testset "count mode accepts one count per window" begin
        res = double_polfed(H, randn(D), (6, 12), exact[70], exact[170]; strategy = :separate)
        @test length(res[1].vals) <= 6 && length(res[2].vals) <= 12
        @test length(res[1].vals) > 0 && length(res[2].vals) > 0
    end

    @testset "rejected input" begin
        x0 = randn(D)
        # Mode is decided by ARITY: a window spec in the count form and vice versa are errors.
        @test_throws Exception double_polfed(H, x0, 10, w1, w2)              # windows, count form
        @test_throws Exception double_polfed(H, x0, exact[70], exact[170])   # points, window form
        @test_throws Exception double_polfed(H, x0, 0, exact[70], exact[170])         # count ≤ 0
        @test_throws Exception double_polfed(H, x0, (5, 0), exact[70], exact[170])    # count ≤ 0
        # One connected region is not a two-window problem.
        @test_throws Exception double_polfed(H, x0, (lo1, hi2), (lo2, hi2))  # overlapping windows
        @test_throws Exception double_polfed(H, x0, w1, w2; strategy = :sideways)
    end
end
