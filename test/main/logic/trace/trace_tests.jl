# The run trace: what a POLFED run says about itself before it turns into iterations.
#
# Ground truth here is *behavioral*, not textual: at each verbosity the run must emit the
# decisions that verbosity promises (level 2 = one line per stage, level 3 = every filter-ladder
# rung, level 0 = nothing at all), and the trace must never change the answer. The few string
# assertions match on stable message fragments, not on whole lines.

using Test
using Polfed
using LinearAlgebra, SparseArrays, Random
using Logging

# Plain globals, not `const`: every test file is `include`d into the same `Main`, and
# common/Formatter_tests.jl and common/logging_tests.jl already bind `C = Polfed.Common`
# there. Declaring it `const` here fails with "cannot declare Main.C constant; it was
# already declared global".
C = Polfed.Common
P = Polfed

# A structured, asymmetric Hermitian fixture (single off-diagonal value, one bit-flip neighbour
# per site) — an interior target on it is non-degenerate, as `polfed` needs.
function _trace_fixture(L::Int = 6; hx = 0.7)
    D = 2^L
    Random.seed!(20260821)
    diagv = randn(D)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        for b in 0:L-1
            push!(I, i); push!(J, xor(i - 1, 1 << b) + 1); push!(V, hx)
        end
    end
    return sparse(I, J, V, D, D)
end

"""Run `f()` at POLFED verbosity `level` and return `(value, log_records)`."""
function _traced(f, level::Int)
    logger = Test.TestLogger(min_level = Logging.Debug)
    value = with_logger(logger) do
        C.with_verbosity(f, level)
    end
    return value, logger.logs
end

_messages(records) = [string(r.message) for r in records]
_has(records, fragment) = any(contains(fragment), _messages(records))

@testset "trace" begin
    H = _trace_fixture()
    D = size(H, 1)
    exact = sort(eigvals(Symmetric(Matrix(H))))
    target = exact[cld(D, 2)] + 0.11
    x0 = (Random.seed!(4); randn(D))

    @testset "verbosity levels select what a run reports" begin
        silent, silent_logs = _traced(() -> P.polfed(H, x0, 4, target), C.POLFED_SILENT_LEVEL)
        info, info_logs = _traced(() -> P.polfed(H, x0, 4, target), C.POLFED_INFO_LEVEL)
        debug, debug_logs = _traced(() -> P.polfed(H, x0, 4, target), C.POLFED_DEBUG_LEVEL)

        @test isempty(silent_logs)

        # Level 2 tells the stage story: mapping → DoS → target → interval → filter → launch → done.
        for fragment in ("polfed started", "Mapping resolved", "density of states",
                         "Target resolved", "Target interval resolved", "Filter resolved",
                         "launching Lanczos", "Lanczos finished")
            @test _has(info_logs, fragment)
        end
        # …but not the per-decision detail.
        @test !_has(info_logs, "filter ladder")
        @test !_has(info_logs, "Result selection")

        # Level 3 adds every filter-ladder rung and the selection rule that produced the answer.
        @test _has(debug_logs, "filter ladder")
        @test _has(debug_logs, "Result selection")
        @test _has(debug_logs, "Lanczos parameters")

        # The trace is inert: the same eigenvalues come back at every level.
        @test collect(silent.vals) ≈ collect(info.vals)
        @test collect(silent.vals) ≈ collect(debug.vals)
    end

    @testset "the pre-Lanczos summary names the filter, the window and the stop rule" begin
        _, logs = _traced(() -> P.polfed(H, x0, 4, target), C.POLFED_INFO_LEVEL)
        launch = only(filter(r -> contains(string(r.message), "launching Lanczos"), logs))
        keys = Dict(launch.kwargs)
        @test contains(keys[:filter], "DeltaUp")                  # which filter was resolved
        @test contains(keys[:filter], "K=")                       # …at which polynomial order
        @test contains(keys[:window], "rescaled")                 # window in BOTH coordinate systems
        @test contains(keys[:window], "in energy")
        @test contains(keys[:eig_target], "NumEigs")              # the Lanczos stop rule
        @test contains(keys[:cost], "H·v per Lanczos iteration")  # the bottleneck, per iteration
    end

    @testset "a rejected ladder rung reports why it was rejected" begin
        # A near-edge window: the sharp first rungs leak past the spectrum boundary and are
        # rejected, so the ladder must both walk on AND say what disqualified each rung.
        _, logs = _traced(C.POLFED_DEBUG_LEVEL) do
            P.resolve_filter(P.Interval(-0.97, -0.90))
        end
        rungs = filter(m -> contains(m, "filter ladder"), _messages(logs))
        @test !isempty(rungs)
        rejected = filter(m -> contains(m, "✗"), rungs)
        # Every rejection carries a reason after the em dash, not just a verdict.
        @test all(m -> contains(m, " — "), rejected)
    end

    @testset "eig-target rendering" begin
        @test P.describe_eig_target(P.LowestEigs(3)) == "Lowest(NumEigs(3))"
        @test P.describe_eig_target(P.HighestEigs(7)) == "Highest(NumEigs(7))"
        window = P.describe_eig_target(P.Highest(P.Window(-0.2, 0.3, 0.14; return_brackets = false)))
        @test contains(window, "Window(-0.2, 0.3")
        @test contains(window, "edge=0.14")
        @test contains(window, "brackets=false")
        @test contains(P.describe_eig_target(P.BothEnds(2, 5)), "BothEnds(low=NumEigs(2)")
    end

    @testset "verbosity is scoped, not sticky" begin
        saved = C.verbosity[]
        try
            C.verbosity[] = C.POLFED_WARN_LEVEL
            # The entrypoint keyword raises the level for one run and restores it afterwards.
            logger = Test.TestLogger(min_level = Logging.Debug)
            with_logger(logger) do
                P.polfed(H, x0, 4, target; verbosity = C.POLFED_INFO_LEVEL)
            end
            @test _has(logger.logs, "launching Lanczos")
            @test C.verbosity[] == C.POLFED_WARN_LEVEL

            # …and it is restored even when the run throws.
            @test_throws ArgumentError C.with_verbosity(C.POLFED_DEBUG_LEVEL) do
                throw(ArgumentError("boom"))
            end
            @test C.verbosity[] == C.POLFED_WARN_LEVEL
        finally
            C.verbosity[] = saved
        end
    end

    @testset "POLFED logging never bypasses the verbosity gate" begin
        # `@warn`/`@info` ignore `verbosity[]`, so level 0 would not silence them. Every POLFED
        # log site must go through `polfed_log` instead — checked at the source, since a missed
        # site only shows up as noise in someone else's run.
        offenders = String[]
        for (root, _, files) in walkdir(joinpath(pkgdir(Polfed), "src")), file in files
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            endswith(path, joinpath("common", "logging.jl")) && continue
            for (n, line) in enumerate(eachline(path))
                code = first(split(line, '#'))                    # ignore comments
                occursin(r"(^|[^`\w])@(warn|info)\s+\"", code) &&
                    push!(offenders, "$path:$n")
            end
        end
        @test isempty(offenders)
    end
end
