using Test
using Polfed
using Polfed: PatchedReport, run_patched_polfed_window_pipeline
using LinearAlgebra, SparseArrays, Random

# QREM: diag(ε) + hx·Σᵢ σᵢˣ — a real-symmetric operator with a full, non-degenerate spectrum.
function _qrem_r(L::Int; hx = 0.7)
    D = 2^L; diagv = randn(D)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        for b in 0:L-1
            push!(I, i); push!(J, xor(i - 1, 1 << b) + 1); push!(V, hx)
        end
    end
    return sparse(I, J, V, D, D)
end

@testset "PatchedReport" begin
    Random.seed!(2027)
    L = 8; D = 2^L
    H = _qrem_r(L)
    f!(Y, X) = mul!(Y, H, X)
    x0 = rand(D); x0 ./= norm(x0)

    saved = Polfed.Common.verbosity[]
    Polfed.Common.verbosity[] = 0
    # Window touches the left edge → one edge peel + several interior patches: exercises both kinds.
    vals, vecs, report = run_patched_polfed_window_pipeline(f!, x0, 12, (:rescaled, (-1.0, 0.4)))
    Polfed.Common.verbosity[] = saved

    @testset "structure carries one record per run" begin
        @test report isa PatchedReport
        @test !isempty(report.patches)
        @test report.total_eigenpairs == length(vals)
        # Every kind is one of the three; at least one edge peel (window touches -1) and interior.
        kinds = [p.kind for p in report.patches]
        @test all(k -> k in (:edge_lowest, :edge_highest, :interior), kinds)
        @test any(==(:edge_lowest), kinds)
        @test any(==(:interior), kinds)
        # The full per-run FactorizationReport is retained (not just aggregates).
        @test all(p -> p.factorization isa Polfed.Numerics.Lanczos.FactorizationReport, report.patches)
    end

    @testset "renders the four sections (plain)" begin
        s = sprint(show, MIME"text/plain"(), report; context = :color => false)
        for header in ("Spectral Decomposition", "Mapping", "Convergence (across", "Timings")
            @test occursin(header, s)
        end
        @test occursin("Runs:", s)                       # decomposition run counts
        @test occursin("H·v", s)                         # aggregate cost line
        @test occursin("Worst eigen residual", s)        # convergence headline
        @test !occursin("Per-run detail", s)             # table is OFF by default
    end

    @testset ":convergence_details appends the per-run table" begin
        s = sprint(show, MIME"text/plain"(), report;
                   context = IOContext(devnull, :color => false, :convergence_details => true))
        @test occursin("Per-run detail", s)
        @test occursin("eigen res", s)                   # a table column header
        @test occursin("edge_lowest", s)                 # a run's kind appears in a row
    end

    @testset "compact show is a one-liner" begin
        @test occursin("PatchedReport", sprint(show, report))
    end
end
