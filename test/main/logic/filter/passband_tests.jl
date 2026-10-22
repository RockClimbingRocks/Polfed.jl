using Test
using UniPolfed: passband, resolve_filter, build_filter, EdgeFilter, DeltaUpFilter, DeltaDownFilter,
              Interval

# The `passband(filter)` accessor is the safety gate's notion of "where the filter is trustworthy":
# a delta filter's resolved effective window; a monotone EdgeFilter's exposed (unbounded) half.

@testset "passband" begin
    @testset "delta filter returns its resolved passband field" begin
        f = resolve_filter(Interval(-0.2, 0.2))          # interior → a delta filter
        @test f isa DeltaUpFilter || f isa DeltaDownFilter
        pb = passband(f)
        @test pb === f.passband                          # the accessor is exactly the field…
        @test pb.left <= -0.2 && pb.right >= 0.2         # …and it covers the requested window
        @test isfinite(pb.left) && isfinite(pb.right)    # a delta passband is bounded
    end

    @testset "EdgeFilter is unbounded toward its exposed edge" begin
        # Left edge (:lowest): unbounded below (the extreme may map just past -1), capped above.
        lo = build_filter(EdgeFilter, Interval(-1.0, -0.9))
        @test passband(lo) == Interval(-Inf, 1.0)

        # Right edge (:highest): capped below, unbounded above.
        hi = build_filter(EdgeFilter, Interval(0.9, 1.0))
        @test passband(hi) == Interval(-1.0, Inf)
    end

    @testset "the passband is the gate the safety check uses" begin
        # A delta bracket sits OUTSIDE the bounded passband → gated out; an EdgeFilter's extreme
        # just past its unbounded edge (e.g. a ground state at -1.0000001) stays in.
        f  = resolve_filter(Interval(-0.2, 0.2))
        pb = passband(f)
        @test !(pb.right + 0.1 <= pb.right)              # a value past the right edge is out-of-band
        @test passband(build_filter(EdgeFilter, Interval(-1.0, -0.9))).left == -Inf
    end
end
