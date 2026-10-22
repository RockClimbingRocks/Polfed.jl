using Test
using UniPolfed
using Random
using Logging

C = UniPolfed.Common

@testset "logging" begin
    @testset "should_log respects the verbosity threshold" begin
        saved = C.verbosity[]
        try
            C.verbosity[] = C.POLFED_WARN_LEVEL
            @test  C.should_log(C.POLFED_WARN_LEVEL)
            @test !C.should_log(C.POLFED_INFO_LEVEL)
            @test !C.should_log(C.POLFED_DEBUG_LEVEL)

            C.verbosity[] = C.POLFED_DEBUG_LEVEL
            @test C.should_log(C.POLFED_WARN_LEVEL)
            @test C.should_log(C.POLFED_INFO_LEVEL)
            @test C.should_log(C.POLFED_DEBUG_LEVEL)
            @test !C.should_log(C.POLFED_DEEP_DEBUG_LEVEL)   # deep-debug is strictly above debug

            C.verbosity[] = C.POLFED_DEEP_DEBUG_LEVEL
            @test C.should_log(C.POLFED_DEEP_DEBUG_LEVEL)
            @test C.should_log(C.POLFED_DEBUG_LEVEL)         # …and subsumes everything below it

            C.verbosity[] = C.POLFED_SILENT_LEVEL
            @test !C.should_log(C.POLFED_WARN_LEVEL)
        finally
            C.verbosity[] = saved
        end
    end

    @testset "polfed_log routes by level and gates by verbosity" begin
        saved = C.verbosity[]
        try
            C.verbosity[] = C.POLFED_DEEP_DEBUG_LEVEL
            @test_logs (:warn, "w") C.polfed_log(C.POLFED_WARN_LEVEL, "w")
            @test_logs (:info, "i") C.polfed_log(C.POLFED_INFO_LEVEL, "i")
            @test_logs (:info, "[DEBUG] d") C.polfed_log(C.POLFED_DEBUG_LEVEL, "d")
            @test_logs (:info, "[DEEP] x") C.polfed_log(C.POLFED_DEEP_DEBUG_LEVEL, "x")

            # A deep-debug message is suppressed at debug verbosity — the whole point of level 4
            # is that the per-iteration firehose stays off until asked for.
            C.verbosity[] = C.POLFED_DEBUG_LEVEL
            @test_logs C.polfed_log(C.POLFED_DEEP_DEBUG_LEVEL, "not shown")

            # below threshold: nothing emitted, returns nothing
            C.verbosity[] = C.POLFED_SILENT_LEVEL
            @test_logs C.polfed_log(C.POLFED_WARN_LEVEL, "silent")
            @test C.polfed_log(C.POLFED_WARN_LEVEL, "silent") === nothing
        finally
            C.verbosity[] = saved
        end
    end

    @testset "kwargs survive the macro splat and reach the log record" begin
        # `@info msg kwargs...` is easy to get subtly wrong; assert the payload actually lands.
        saved = C.verbosity[]
        try
            C.verbosity[] = C.POLFED_INFO_LEVEL
            rec = let r = nothing
                Test.collect_test_logs() do
                    C.polfed_log(C.POLFED_INFO_LEVEL, "msg", order = 42, filter = :DeltaUpFilter)
                end[1][1]
            end
            @test rec.message == "msg"
            @test rec.kwargs[:order] == 42
            @test rec.kwargs[:filter] === :DeltaUpFilter
        finally
            C.verbosity[] = saved
        end
    end

    @testset "the pipeline is silent at level 0 and progressively louder above it" begin
        # The contract users actually rely on: raising `verbosity[]` must strictly ADD records.
        # This is a regression guard for ungated `@warn`s, which ignore `verbosity` entirely and
        # so would leak output at level 0.
        saved = C.verbosity[]
        try
            # QREM fixture (same shape as the entrypoint tests): diag(ε) + hx·Σᵢ σᵢˣ.
            L = 6; D = 2^L
            Random.seed!(20260721)
            diagv = randn(D)
            H = zeros(D, D)
            for i in 1:D
                H[i, i] = diagv[i]
                for b in 0:L-1
                    H[i, xor(i - 1, 1 << b) + 1] = 0.7
                end
            end
            x0 = randn(D)

            counts = map(0:4) do v
                C.verbosity[] = v
                logs, _ = Test.collect_test_logs(min_level = Logging.Debug) do
                    UniPolfed.polfed(H, copy(x0), 4, (:unrescaled, 0.0))
                end
                length(logs)
            end
            @test counts[1] == 0                     # level 0 is genuinely silent
            @test issorted(counts)                   # each level only adds
            @test counts[3] > counts[2]              # INFO says something
            @test counts[5] > counts[4]              # DEEP adds per-iteration tracing
        finally
            C.verbosity[] = saved
        end
    end
end
