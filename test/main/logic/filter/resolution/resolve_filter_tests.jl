
import Polfed: resolve_filter, Interval, DeltaUpFilter, DeltaDownFilter, DoubleDeltaFilter,
               EdgeFilter, validate_filter, exposed_end, _cutoff_values, _build_candidate,
               _first_valid, Defaults

# The coverage guarantee validate_filter now carries for the deltas, asserted explicitly.
_pb_covers(f, iv) = f.passband.left <= iv.left + 1e-8 && f.passband.right >= iv.right - 1e-8

@testset "resolve_filter" begin
    @testset "per-filter ladders (the resolution interface)" begin
        # Each filter type declares its OWN ladder — tested by intrinsic properties, not by a
        # hardcoded first value or cross-filter equality (the ladders are independently tunable).
        up = collect(_cutoff_values(DeltaUpFilter))
        @test issorted(up) && all(c -> 0 < c < 1, up)               # positive, ascending (sharpest first)
        @test all(≈(0.01), diff(up))                                # uniform 0.01 step
        down = collect(_cutoff_values(DeltaDownFilter))
        @test all(<(0), down) && issorted(down; rev = true)         # the -1 peak: negative cutoffs, descending
        @test all(c -> 0 < c < 1, collect(_cutoff_values(DoubleDeltaFilter)))  # magnitudes, used as ∓c
        @test length(_cutoff_values(EdgeFilter)) == 1               # fixed order, single candidate
    end

    @testset "interior window → valid DeltaUpFilter (a ladder rung)" begin
        iv = Interval(-0.1, 0.1)
        f = resolve_filter(iv)
        @test f isa DeltaUpFilter
        @test f.cutoff in _cutoff_values(DeltaUpFilter)             # the winner is one of the ladder rungs
        @test f.order > 0
        @test exposed_end(f) === :highest
        @test validate_filter(f)
        @test _pb_covers(f, iv)                                     # passband covers the request…
        @test f.passband.left > -1 && f.passband.right < 1          # …and is strictly interior
    end

    @testset "near-edge interior window stays a delta case" begin
        # The user-facing contract: a window CLOSE to the edge but not touching it is served by
        # a delta (never escalated to an EdgeFilter), possibly on a higher ladder rung.
        iv = Interval(-0.99, -0.9)
        f = resolve_filter(iv)
        @test f isa DeltaUpFilter
        @test validate_filter(f) && _pb_covers(f, iv)
        @test f.passband.left > -1.0 + 1e-8                         # the winning rung stays off the edge

        # …and one where the FIRST rung demonstrably fails, so the ladder must retry:
        iv2 = Interval(-0.95, -0.9)
        f2 = resolve_filter(iv2)
        @test f2 isa DeltaUpFilter
        @test f2.cutoff > first(_cutoff_values(DeltaUpFilter))      # the first (sharpest) rung fails here
        @test validate_filter(f2) && _pb_covers(f2, iv2)
        @test f2.passband.left > -1.0 + 1e-8
    end

    @testset "chain mechanics: every member works standalone via _first_valid" begin
        @test DeltaUpFilter in Defaults.filters.INTERIOR_FILTER_CHAIN && DeltaDownFilter in Defaults.filters.INTERIOR_FILTER_CHAIN
        @test EdgeFilter in Defaults.filters.EDGE_FILTER_CHAIN
        # the second chain member is a real fallback: it resolves an interior window on its own
        iv = Interval(-0.1, 0.1)
        fd = _first_valid(DeltaDownFilter, iv)
        @test fd isa DeltaDownFilter
        @test fd.cutoff == first(_cutoff_values(DeltaDownFilter))
        @test validate_filter(fd) && _pb_covers(fd, iv)
        # a hopeless rung: _build_candidate builds it, validate_filter rejects it (the sharpest
        # rung leaks past the near-edge window's boundary)
        bad = _build_candidate(DeltaUpFilter, Interval(-0.95, -0.9), first(_cutoff_values(DeltaUpFilter)))
        @test bad isa DeltaUpFilter && !validate_filter(bad)
    end

    @testset "window touching an edge → EdgeFilter" begin
        f = resolve_filter(Interval(-1.0, -0.9))
        @test f isa EdgeFilter && exposed_end(f) === :lowest
        f2 = resolve_filter(Interval(0.9, 1.0))
        @test f2 isa EdgeFilter && exposed_end(f2) === :highest
    end

    @testset "invalid requests throw" begin
        @test_throws ArgumentError resolve_filter(Interval(-1.0, 1.0))       # full spectrum
        @test_throws ArgumentError resolve_filter(Interval(0.5, 0.2))        # malformed
        # a window no delta can serve (degenerates to the constant filter): the WHOLE interior
        # chain — DeltaUp then DeltaDown — exhausts and errors informatively
        @test_throws ErrorException resolve_filter(Interval(-0.99, 0.99))
    end

    @testset "two interior windows → one DoubleDeltaFilter (1-tuple)" begin
        iv1, iv2 = Interval(-0.5, -0.4), Interval(0.4, 0.5)
        result = resolve_filter([iv1, iv2])
        @test result isa Tuple && length(result) == 1
        d = result[1]
        @test d isa DoubleDeltaFilter
        @test validate_filter(d)                                    # incl. per-window coverage
        @test exposed_end(d) === :both
    end

    @testset "pair containing a spectrum edge → independent filters (2-tuple)" begin
        result = resolve_filter([Interval(-1.0, -0.9), Interval(0.2, 0.3)])
        @test result isa Tuple && length(result) == 2
        @test result[1] isa EdgeFilter && exposed_end(result[1]) === :lowest
        @test result[2] isa DeltaUpFilter
    end

    @testset "pair validation" begin
        @test_throws ArgumentError resolve_filter([Interval(-0.1, 0.1)])                        # not two
        @test_throws ArgumentError resolve_filter(
            [Interval(-0.1, 0.1), Interval(0.2, 0.3), Interval(0.5, 0.6)])                      # not two
        @test_throws ArgumentError resolve_filter([Interval(0.2, 0.3), Interval(-0.5, -0.4)])   # unordered
        @test_throws ArgumentError resolve_filter([Interval(-0.5, 0.1), Interval(0.0, 0.5)])    # overlap
    end
end
