# PolfedResult: iteration/indexing protocol, and the post-processing guarantees from
# build_report.jl — eigenvalues filtered to the target interval and sorted ascending. Uses the
# shared fixture (report_fixture.jl).

using LinearAlgebra
using UniPolfed: PolfedResult, MappingConfig, polfed, eigenvector, eigenpair

@testset "PolfedResult" begin
    @testset "destructures as (vals, vecs, report), like LanczosResult" begin
        @test RES isa PolfedResult
        @test length(RES.vals) >= 1

        vals, vecs, report = RES
        @test vals === RES.vals
        @test vecs === RES.vecs
        @test report === RES.report

        # exactly three items, in that order
        @test length(collect(RES)) == 3
    end

    @testset "layout-aware eigenvector / eigenpair access" begin
        for i in eachindex(RES.vals)
            v = eigenvector(RES, i)
            @test length(v) == REPORT_D
            @test norm(v) ≈ 1 atol = 1e-6            # eigenvectors are normalized

            λ, v2 = eigenpair(RES, i)
            @test λ === RES.vals[i]
            @test v2 == v
        end
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
        @test RES.converged >= length(RES.vals)  # total converged ≥ in-interval returned
    end
end
