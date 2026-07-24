using Test
using UniPolfed: DoSResult, Interval, integrate_chebyshev_weighted, intervalize_interval,
              intervalize_interval_overlapping

"""
A deterministic DoS with `total_states` over `[-1, 1]`. With only the zeroth
Chebyshev coefficient present, interval counts are analytic and
`resolve_interval` can size patches without stochastic KPM noise.
"""
function _uniform_patch_dos(total_states::Real=1000)
    coefficients = [Float64(total_states)]
    moments = copy(coefficients)
    density = x -> Float64(total_states) / (π * sqrt(1 - x^2))
    return DoSResult(moments, coefficients, density)
end

_patch_state_count(dos::DoSResult, interval::Interval) =
    integrate_chebyshev_weighted(dos.coefficients, interval.left, interval.right) / π

function _test_valid_partition(
    intervals,
    full_iv::Interval,
    dos::DoSResult,
    howmany::Int;
    edge_distance::Real,
)
    @test !isempty(intervals)
    @test all(iv -> isfinite(iv.left) && isfinite(iv.right), intervals)
    @test all(iv -> iv.left < iv.right, intervals)
    @test first(intervals).left == full_iv.left
    @test last(intervals).right == full_iv.right
    @test all(iv -> full_iv.left <= iv.left < iv.right <= full_iv.right, intervals)
    @test all(
        intervals[i].right == intervals[i + 1].left
        for i in 1:length(intervals)-1
    )

    left_edge = -1.0 + edge_distance
    right_edge = 1.0 - edge_distance
    interior_stop = min(full_iv.right, right_edge)

    if full_iv.right <= left_edge || full_iv.left >= right_edge
        @test length(intervals) == 1
    end
    if full_iv.left < left_edge < full_iv.right
        @test first(intervals).right == left_edge
    end
    if full_iv.left < right_edge < full_iv.right
        @test last(intervals).left == right_edge
    end

    # Edge pieces are intentionally not count-sized. Every complete interior
    # piece contains the requested count; only the clipped final piece may hold
    # fewer states.
    for interval in intervals
        inside_interior = left_edge <= interval.left && interval.right <= right_edge
        inside_interior || continue

        count = _patch_state_count(dos, interval)
        @test count > 0
        if interval.right < interior_stop
            @test count ≈ howmany rtol=1e-5
        else
            @test count <= howmany * (1 + 1e-5)
        end
    end
    return nothing
end

@testset "intervalize_interval" begin
    dos = _uniform_patch_dos()
    howmany = 100
    edge_distance = 0.2

    cases = [
        "narrow interior"           =>  Interval(-0.10, -0.09),
        "interior centered"         =>  Interval(-0.20, 0.20),
        "interior several pieces"   =>  Interval(-0.70, 0.70),
        "starts at left boundary"   =>  Interval(-0.80, 0.40),
        "ends at right boundary"    =>  Interval(0.20, 0.80),
        "left edge only"            =>  Interval(-1.00, -0.80),
        "inside left edge"          =>  Interval(-0.95, -0.85),
        "left edge plus interior"   =>  Interval(-1.00, 0.30),
        "right edge only"           =>  Interval(0.80, 1.00),
        "inside right edge"         =>  Interval(0.85, 0.95),
        "interior plus right edge"  =>  Interval(-0.30, 1.00),
        "both edges"                =>  Interval(-1.00, 1.00),
    ]

    @testset "$name" for (name, full_iv) in cases
        intervals = intervalize_interval(
            dos,
            full_iv,
            howmany;
            edge_distance=edge_distance,
        )
        _test_valid_partition(
            intervals,
            full_iv,
            dos,
            howmany;
            edge_distance=edge_distance,
        )
    end

    @testset "interior patches have the requested DoS count" begin
        full_iv = Interval(-0.70, 0.70)
        intervals = intervalize_interval(
            dos,
            full_iv,
            howmany;
            edge_distance=edge_distance,
        )
        counts = _patch_state_count.(Ref(dos), intervals)

        @test length(intervals) > 1
        @test all(
            count -> isapprox(count, howmany; rtol=1e-5),
            counts[1:end-1],
        )
        @test 0 < last(counts) <= howmany * (1 + 1e-5)
    end

    @testset "zero edge distance partitions the full spectrum" begin
        full_iv = Interval(-1.0, 1.0)
        intervals = intervalize_interval(dos, full_iv, howmany; edge_distance=0)
        _test_valid_partition(
            intervals,
            full_iv,
            dos,
            howmany;
            edge_distance=0,
        )
        @test length(intervals) > 1
    end

    @testset "accepts general Real and Integer inputs" begin
        intervals = intervalize_interval(
            dos,
            Interval(-0.5, 0.5),
            Int32(howmany);
            edge_distance=Float32(edge_distance),
        )
        @test first(intervals).left == -0.5
        @test last(intervals).right == 0.5
    end

    @testset "invalid inputs are rejected" begin
        valid = Interval(-0.5, 0.5)
        @test_throws ArgumentError intervalize_interval(dos, valid, 0)
        @test_throws ArgumentError intervalize_interval(dos, valid, -1)

        for distance in (-0.1, 1.0, 1.1, Inf, NaN)
            @test_throws ArgumentError intervalize_interval(
                dos,
                valid,
                howmany;
                edge_distance=distance,
            )
        end

        invalid_intervals = [
            Interval(-1.0, -1.0),
            Interval(0.2, -0.2),
            Interval(-1.1, 0.0),
            Interval(0.0, 1.1),
            Interval(NaN, 0.2),
            Interval(-0.2, Inf),
        ]
        for interval in invalid_intervals
            @test_throws ArgumentError intervalize_interval(dos, interval, howmany)
        end
    end
end

@testset "intervalize_interval_overlapping" begin
    dos = _uniform_patch_dos()
    howmany = 100
    edge_distance = 0.2

    @testset "returns the base partition as `unpadded`" begin
        full_iv = Interval(-0.6, 0.6)
        base = intervalize_interval(dos, full_iv, howmany; edge_distance=edge_distance)
        res  = intervalize_interval_overlapping(dos, full_iv, howmany, 20; edge_distance=edge_distance)
        @test res.unpadded == base                       # exposes the pre-pad patches verbatim
        @test length(res.padded) == length(res.unpadded) # …aligned index-for-index with the padded ones
    end

    @testset "pad == 0 makes `padded` identical to `unpadded`" begin
        full_iv = Interval(-0.6, 0.6)
        res = intervalize_interval_overlapping(dos, full_iv, howmany, 0; edge_distance=edge_distance)
        @test res.padded == res.unpadded
    end

    @testset "each side's pad band holds ~howmany_pad states, and patches grow outward" begin
        # Interior request so no pad band is clamped to a spectrum edge: every pad band holds
        # EXACTLY howmany_pad states (uniform analytic DoS), and this is the count that matters.
        full_iv = Interval(-0.5, 0.5)
        pad = 30
        res = intervalize_interval_overlapping(dos, full_iv, howmany, pad; edge_distance=edge_distance)

        for (b, o) in zip(res.unpadded, res.padded)
            @test o.left  < b.left               # grown outward on the left…
            @test o.right > b.right              # …and on the right
            # The state count of the band added on each side is exactly the requested pad.
            @test _patch_state_count(dos, Interval(o.left, b.left))  ≈ pad rtol=1e-5
            @test _patch_state_count(dos, Interval(b.right, o.right)) ≈ pad rtol=1e-5
        end
    end

    @testset "adjacent padded patches genuinely overlap" begin
        full_iv = Interval(-0.6, 0.6)
        padded = intervalize_interval_overlapping(dos, full_iv, howmany, 20; edge_distance=edge_distance).padded
        @test length(padded) > 1
        # A base partition only touches at endpoints; the padded one strictly overlaps.
        for i in 1:length(padded)-1
            @test padded[i].right > padded[i+1].left
        end
        # …and every patch is still a well-formed, in-spectrum interval.
        @test all(iv -> -1.0 <= iv.left < iv.right <= 1.0, padded)
    end

    @testset "padding clamps to the spectrum edges, never past ±1" begin
        # A request reaching ±1: the outward pad on the boundary side cannot grow past the edge.
        padded = intervalize_interval_overlapping(dos, Interval(-1.0, 1.0), howmany, 40; edge_distance=edge_distance).padded
        @test all(iv -> -1.0 <= iv.left < iv.right <= 1.0, padded)
        @test first(padded).left == -1.0         # left pad of the leftmost patch is clamped
        @test last(padded).right ==  1.0         # right pad of the rightmost patch is clamped
    end

    @testset "negative pad is rejected" begin
        @test_throws ArgumentError intervalize_interval_overlapping(dos, Interval(-0.5, 0.5), howmany, -1)
    end
end
