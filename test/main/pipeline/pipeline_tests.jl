using LinearAlgebra
using Random
using Test
using UniPolfed: DoSConfig, MappingConfig, full_polfed, patched_polfed

@testset "edge-first patched and full pipelines" begin
    Random.seed!(2407)
    dim = 24
    exact = collect(range(-1.0, 1.0; length = dim))
    H = Diagonal(exact)
    f!(Y, X) = mul!(Y, H, X)
    x0 = normalize(randn(dim))
    mc = MappingConfig(Emin = -1.0, Emax = 1.0)
    dosc = DoSConfig(N = 80, R = 200, kernel = :Jackson)
    common = (
        mc = mc, dosc = dosc, tol = 1e-13, eigentol = 1e-8,
        edge_distance = 0.2, howmany_pad = 2, max_retries = 2,
        retry_growth = 1.5,
    )

    window_vals, window_vecs, window_report =
        patched_polfed(
            f!, x0, 5, (:rescaled, (-0.95, 1.0)); common...,
        )
    expected_window = filter(value -> -0.95 <= value <= 1.0, exact)
    @test window_vals ≈ expected_window atol = 2e-8
    @test length(window_vecs) == length(window_vals)
    @test count(patch -> patch.kind == :edge_both, window_report.patches) == 1
    @test any(patch -> patch.kind == :interior, window_report.patches)

    target_vals, target_vecs, target_report = patched_polfed(
        f!, x0, 4, 5, (:rescaled, 0.0); common...,
    )
    nearest = sort(exact[partialsortperm(abs.(exact), 1:4)])
    @test target_vals ≈ nearest atol = 2e-8
    @test size(target_vecs) == (dim, 4)
    @test target_report.total_eigenpairs == 4

    full_vals, full_vecs, full_report = full_polfed(
        H, x0, 5; common...,
    )
    @test full_vals ≈ exact atol = 2e-8
    @test size(full_vecs) == (dim, dim)
    @test full_report.mapping.layout == "RowLayout"
    @test count(patch -> patch.kind == :edge_both, full_report.patches) == 1
end
