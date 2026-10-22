# StageTimer / @stage / @stages: name-addressed, self-accumulating, ambient stage timing.
# The contract under test:
#   • first `@stage :x` CREATES the entry, every later one ADDS to it (wall, cpu, and hit count)
#   • `@stage` returns the timed expression's VALUE (so `y = @stage :x f()` reads normally)
#   • the timer is AMBIENT — a `@stage` inside a called function records into the caller's `@stages`
#   • with no `@stages` scope open, `@stage` is a pass-through (timing is opt-in)
#   • nested stages form a tree; children keep insertion (execution) order
#   • the ambient scope is restored even when the timed expression throws

using Test
using Polfed.Common: StageTimer, @stage, @stages, stage_names, total_wall, total_cpu,
                     _stage_current, _elapsed, _cpu_time_ns

@testset "StageTimer" begin

    @testset "accumulation: create on first hit, add on every later one" begin
        t = @stages begin
            for _ in 1:3
                @stage :work sum(abs2, 1:10_000)
            end
        end
        @test stage_names(t) == [:work]
        @test t[:work, :n] == 3                    # three hits folded into ONE entry
        @test t[:work] > 0.0                       # wall accumulated
        @test t[:work, :cpu] >= 0.0
        @test t[:work] == t[:work, :wall]          # the bare index is wall time
    end

    @testset "@stage returns the expression's value" begin
        local y
        t = @stages begin
            y = @stage :compute (2 + 3)
        end
        @test y == 5                               # value passes through the macro…
        @test t[:compute, :n] == 1                 # …and the stage was still recorded
    end

    @testset "@stages introduces a scope (like try): pre-declare what you need after" begin
        # A variable declared OUTSIDE is rebound by an assignment inside — this is what
        # `lanczos_algorithm` relies on for its `result`.
        outer = nothing
        t = @stages begin
            outer = 41 + 1
        end
        @test outer == 42
        @test t isa StageTimer
    end

    @testset "ambient: a stage inside a called function records into the caller's scope" begin
        inner() = @stage :inner sum(1:1000)        # no timing parameter anywhere
        t = @stages begin
            @stage :outer inner()
        end
        @test stage_names(t) == [:outer]           # :inner is NOT a sibling…
        node = t[:outer, :node]
        @test stage_names(node) == [:inner]        # …it nests as a child of :outer
        @test node[:inner, :n] == 1
    end

    @testset "no scope open → pass-through, untimed" begin
        f() = @stage :never_recorded 7
        @test _stage_current() === nothing         # nothing ambient here
        @test f() == 7                             # still evaluates and returns
    end

    @testset "insertion order is execution order, not alphabetical" begin
        t = @stages begin
            @stage :zebra 1
            @stage :alpha 2
            @stage :zebra 3                        # revisiting must NOT reorder
            @stage :mid   4
        end
        @test stage_names(t) == [:zebra, :alpha, :mid]
        @test t[:zebra, :n] == 2
    end

    @testset "missing stages read as zero (no haskey guards needed in report code)" begin
        t = @stages (@stage :only 1)
        @test !haskey(t, :absent)
        @test t[:absent] == 0.0
        @test t[:absent, :cpu] == 0.0
        @test t[:absent, :n] == 0
        @test t[:absent, :node] === nothing
        @test_throws ArgumentError t[:only, :bogus]
    end

    @testset "_elapsed: a backwards clock reads 0, never 2^64" begin
        # Both clocks return UInt64, so an inverted pair would wrap to ~1.8e13 s and swamp every
        # other stage rather than reading as the ~0 it is.
        @test _elapsed(UInt64(10), UInt64(4), 1e-6) ≈ 6e-6      # normal forward case
        @test _elapsed(UInt64(4), UInt64(10), 1e-6) == 0.0      # inverted → floored, NOT 1.8e13
        @test _elapsed(UInt64(0), UInt64(1), 1e-9) == 0.0       # one-tick inversion
        @test _elapsed(UInt64(7), UInt64(7), 1e-6) == 0.0       # flat clock
        # The guard must be an unsigned COMPARISON, not a widening cast: `time_ns()` counts since
        # boot and outgrows typemax(Int64), which `Int64(stop)` would reject with InexactError.
        big = typemax(UInt64) - UInt64(5)
        @test _elapsed(big, big - UInt64(1000), 1e-9) ≈ 1e-6
        @test _elapsed(big - UInt64(1000), big, 1e-9) == 0.0

        # …and end to end: no stage may report a physically absurd CPU time.
        t = @stages begin
            for _ in 1:200
                @stage :tiny 1 + 1                              # short enough to hit a flat clock
            end
        end
        @test t[:tiny, :cpu] >= 0.0
        @test t[:tiny, :cpu] < 60.0                             # a trivial loop, not geological time
        @test total_cpu(t) < 60.0
    end

    @testset "_cpu_time_ns is a real, non-decreasing process CPU clock" begin
        # This replaced CPUTime.jl's `CPUtime_us`, which read an UNINITIALIZED malloc'd rusage
        # buffer without checking uv_getrusage's status and multiplied the stale bytes by 1e6 —
        # it reported ~430 000 years of CPU for a stage that used 0.36 s of wall time.
        a = _cpu_time_ns()
        acc = 0.0
        for i in 1:4_000_000                                    # burn a measurable slice of CPU
            acc += sqrt(i)
        end
        b = _cpu_time_ns()
        @test acc > 0                                           # keep the loop from being elided
        @test b >= a                                            # process CPU time never goes back
        used = (b - a) * 1e-9
        @test used > 0.0                                        # the work IS observed…
        @test used < 60.0                                       # …and is not fictional
    end

    @testset "a timed stage's CPU time is physically consistent with its wall time" begin
        t = StageTimer()
        acc = @stage t :burn begin
            s = 0.0
            for i in 1:4_000_000
                s += sqrt(i)
            end
            s
        end
        @test acc > 0
        # Single-threaded work: CPU ≈ wall, and can never be wildly beyond it. The upper bound is
        # generous (thread count) but still catches a garbage clock by many orders of magnitude.
        @test t[:burn, :cpu] > 0.0
        @test t[:burn, :cpu] <= t[:burn] * max(2, Threads.nthreads()) + 1.0
    end

    @testset "totals sum the direct children" begin
        t = @stages begin
            @stage :a sum(abs2, 1:5_000)
            @stage :b sum(abs2, 1:5_000)
        end
        @test total_wall(t) ≈ t[:a] + t[:b]
        @test total_cpu(t)  ≈ t[:a, :cpu] + t[:b, :cpu]
        @test total_wall(StageTimer()) == 0.0      # empty timer is well-defined
    end

    @testset "a throwing stage still records and restores the ambient scope" begin
        t = StageTimer()
        outer = @stages begin
            @test_throws ErrorException (@stage :boom error("kaboom"))
            @stage :after 1                        # scope survived the throw…
        end
        @test outer[:boom, :n] == 1                # …and the failed stage was still recorded
        @test outer[:after, :n] == 1
        @test _stage_current() === nothing         # ambient fully unwound afterwards
    end

    @testset "explicit 3-arg form records into a given timer" begin
        t = StageTimer()
        v = @stage t :explicit (10 * 2)
        @test v == 20
        @test t[:explicit, :n] == 1
        @test stage_names(t) == [:explicit]
    end

    @testset "nested @stages scopes do not leak into each other" begin
        inner_t = nothing
        outer_t = @stages begin
            @stage :outer_only 1
            inner_t = @stages (@stage :inner_only 2)
        end
        @test stage_names(outer_t) == [:outer_only]
        @test stage_names(inner_t) == [:inner_only]
        @test _stage_current() === nothing
    end

    @testset "show renders the tree" begin
        t = @stages begin
            @stage :parent (@stage :child 1)
        end
        s = sprint(show, MIME"text/plain"(), t)
        @test occursin("parent", s) && occursin("child", s)
        @test occursin("StageTimer", sprint(show, t))
    end
end
