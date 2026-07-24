# PolfedResult: iteration/indexing protocol, and the post-processing guarantees from
# build_report.jl — eigenvalues filtered to the target interval and sorted ascending. Uses the
# shared fixture (report_fixture.jl).

using LinearAlgebra
using UniPolfed: PolfedResult, MappingConfig, polfed

@testset "PolfedResult" begin
    @testset "iteration / indexing protocol" begin
        @test RES isa PolfedResult
        @test length(RES) == length(RES.vals)
        @test length(RES) >= 1

        # iterate → (val, vec) pairs; vec is a length-dim vector view.
        pairs = collect(RES)
        @test length(pairs) == length(RES)
        for (λ, v) in pairs
            @test λ isa Real
            @test length(v) == REPORT_D
            @test norm(v) ≈ 1 atol = 1e-6            # eigenvectors are normalized
        end

        # indexing agrees with iteration; firstindex/lastindex.
        @test RES[1] == pairs[1]
        @test RES[lastindex(RES)] == pairs[end]
        @test firstindex(RES) == 1
    end

    @testset "sorted ascending by true energy" begin
        @test issorted(RES.vals)
        @test length(RES.residuals) == length(RES.vals)
    end

    @testset "filtered to the target interval, matching exact diagonalization" begin
        el = RES.report.transform.energy_left
        er = RES.report.transform.energy_right
        # every returned value is inside the interval and a genuine eigenvalue of H
        for λ in RES.vals
            @test el - 1e-6 <= λ <= er + 1e-6
            @test minimum(abs.(REPORT_EXACT .- λ)) < 1e-6
        end
        # the returned set equals the exact eigenvalues in that interval (nearest-howmany)
        k = min(length(RES.vals), REPORT_HOWMANY)
        near_exact = sort(sort(REPORT_EXACT; by = λ -> abs(λ - REPORT_TARGET))[1:k])
        @test maximum(abs.(sort(collect(RES.vals))[1:k] .- near_exact)) < 1e-6
    end

    @testset "converged counts total; length is the in-interval count" begin
        @test RES.converged >= length(RES)       # total converged ≥ in-interval returned
    end
end
