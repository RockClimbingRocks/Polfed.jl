# End-to-end tests for spectrum-EDGE targets through `polfed` — the paths that auto-pick the
# EdgeFilter. Ground truth is exact diagonalization. Covered end to end:
#   • window mode with a bound past a spectrum edge (clamped to ±1): exercises the EdgeFilter,
#     the :Taylor (Horner) transform, the Window criterion's infinite-bound bracket waiver, and
#     the edge-aware result trim — the returned set must be EXACTLY the ED eigenvalues in the
#     window, including the extremal one sitting on the mapping's Emin/Emax estimate;
#   • count mode targeting an extremal energy: the grown interval reaches the edge, so the
#     EdgeFilter is picked and the `howmany` edge eigenpairs must match ED;
#   • the :both filter driven directly (build_transform + BothEnds Lanczos): both spectrum edges
#     harvested from ONE Krylov space (no `polfed` path for :both yet — multi-window TODO).

using LinearAlgebra, SparseArrays, Random, Logging, Test
using UniPolfed: polfed, MappingConfig, PolfedResult
import UniPolfed: EdgeFilter, Interval, build_filter, build_transform, build_mapping_plan,
               resolve_eig_target, exposed_end
import UniPolfed.Numerics.Lanczos: lanczos

# QREM: diag(ε) + hx·Σᵢ σᵢˣ (see matrix_entrypoint_tests / qrem_fixture). Real-symmetric.
function _qrem_e(L::Int; hx = 0.7, T = Float64)
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

@testset "edge targets through polfed (EdgeFilter e2e)" begin
    Random.seed!(20260717)
    L = 8; D = 2^L
    H = _qrem_e(L)
    exact = sort(eigvals(Symmetric(Matrix(H))))

    # Run capturing logs; assert the EdgeFilter was the one picked and the completeness
    # certificate held (no INCOMPLETE warning — the infinite-bound side needs no bracket).
    function run_edge(f)
        logger = Test.TestLogger()
        res = with_logger(f, logger)
        @test res isa PolfedResult
        @test res.report.transform.filter_name == "EdgeFilter"
        @test !any(r -> occursin("INCOMPLETE", r.message), logger.logs)
        return res
    end

    # An edge window has only ONE finite bound, so it comes back with a single bracket — the side
    # running past the spectrum extreme has nothing beyond it to bracket. `EdgeFilter` is a monotone
    # monomial power that cannot leak, so unlike a delta filter its bracket IS the true immediate
    # neighbour, and is asserted as such here.
    @testset "window past the LOW edge returns the ED set + one bracket" begin
        n = 8
        hi = (exact[n] + exact[n+1]) / 2                        # bound in a spectral gap
        res = run_edge(() -> polfed(H, randn(D), (exact[1] - 1.0, hi)))
        vals = sort(collect(res.vals))
        want = exact[1:n+1]                                     # edge eigenvalue NOT trimmed, + bracket
        @test length(vals) == n + 1
        @test length(vals) == n + 1 && maximum(abs.(vals .- want)) < 1e-8
    end

    @testset "window past the HIGH edge, optimized :Taylor row kernel" begin
        n = 6
        lo = (exact[end-n] + exact[end-n+1]) / 2
        res = run_edge(() -> polfed(H, randn(D), (lo, exact[end] + 1.0);
                                    mc = MappingConfig(optimize_mapping = :shuf)))
        vals = sort(collect(res.vals))
        want = exact[end-n:end]                                 # + one bracket below
        @test length(vals) == n + 1
        @test length(vals) == n + 1 && maximum(abs.(vals .- want)) < 1e-8
    end

    @testset "count mode at the $(name) edge" for (name, target_E, pick) in
            (("low", exact[1], v -> v[1:6]), ("high", exact[end], v -> v[end-5:end]))
        res = run_edge(() -> polfed(H, randn(D), 6, target_E))
        vals = sort(collect(res.vals))
        @test length(vals) >= 6
        @test maximum(abs.(pick(vals) .- pick(exact))) < 1e-8
    end

    @testset ":both — both edges from one Krylov space (direct transform)" begin
        f! = (Y, X) -> mul!(Y, H, X)
        mp = build_mapping_plan(MappingConfig(), f!, randn(D))
        filter = build_filter(EdgeFilter, Interval(-1.0, -0.9), Interval(0.9, 1.0); order = 5)
        f!_t = build_transform(filter, mp)
        res = lanczos(f!_t, mp.sys.x0, resolve_eig_target(exposed_end(filter), 4);
                      maxdim = 120, mapvals = mp.f!)
        @test res.converged >= 8
        v = sort(collect(res.vals[1:res.converged]))
        @test maximum(abs.(v[1:4] .- exact[1:4])) < 1e-8            # 4 lowest…
        @test maximum(abs.(v[end-3:end] .- exact[end-3:end])) < 1e-8 # …and 4 highest, one run
    end
end
