# The four-method end-spec interface — the whole "what does convergence mean" surface.
# One section per spec. These are pure functions of the Ritz values `p` (transformed) and,
# for Window only, the mapped values `E`, so every case is exercised directly.
#
# Walk order is ALWAYS extreme-first: `1:n` at the low end, `n:-1:1` at the high end. Every
# rule below is a statement about a prefix of that order.

using Test
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

@testset "end_spec" begin
    p = [-3.0, -1.0, 0.0, 2.0, 5.0]        # ascending Ritz values

    @testset "end_walk / past_boundary — the direction primitives" begin
        @test collect(LZ.end_walk(5, true))  == [1, 2, 3, 4, 5]
        @test collect(LZ.end_walk(5, false)) == [5, 4, 3, 2, 1]

        # Walking inward, p RISES from the low end and FALLS from the high end, so "past"
        # flips sense with `islow`.
        @test  LZ.past_boundary(0.0,  1.0, true)     # low walk has climbed past 0
        @test !LZ.past_boundary(0.0, -1.0, true)
        @test  LZ.past_boundary(0.0, -1.0, false)    # high walk has fallen past 0
        @test !LZ.past_boundary(0.0,  1.0, false)
        # Exactly ON the boundary is NOT past it — the guard must be strictly beyond.
        @test !LZ.past_boundary(0.0, 0.0, true)
        @test !LZ.past_boundary(0.0, 0.0, false)
    end

    # ── NumEigs ───────────────────────────────────────────────────────────────
    @testset "NumEigs" begin
        @test LZ.pool_indices(NumEigs(2), p, true)  == [1, 2]
        @test LZ.pool_indices(NumEigs(2), p, false) == [5, 4]
        @test LZ.pool_indices(NumEigs(99), p, true) == [1, 2, 3, 4, 5]   # capped at n

        @test !LZ.needs_mapped(NumEigs(2))

        # No mapped values are consulted — the rule is pure run length.
        @test !LZ.is_satisfied(NumEigs(2), p[1:1], Float64[], true)
        @test  LZ.is_satisfied(NumEigs(2), p[1:2], Float64[], true)
        @test !LZ.is_satisfied(NumEigs(3), p[1:2], Float64[], true)

        @test LZ.select(NumEigs(2), [1, 2], p[1:2], Float64[], true) == [1, 2]
    end

    # ── ToValue ───────────────────────────────────────────────────────────────
    @testset "ToValue" begin
        @testset "pool = everything inside the bound + one guard past it" begin
            # low end, bound 0.0: p ≤ 0 is {1,2,3}, guard adds index 4.
            @test LZ.pool_indices(ToValue(0.0), p, true)  == [1, 2, 3, 4]
            # high end, bound 0.0: p ≥ 0 is {5,4,3}, guard adds index 2.
            @test LZ.pool_indices(ToValue(0.0), p, false) == [5, 4, 3, 2]
            # a bound past the whole spectrum cannot over-run the array
            @test LZ.pool_indices(ToValue(99.0), p, true) == [1, 2, 3, 4, 5]
        end

        @test !LZ.needs_mapped(ToValue(0.0))

        @testset "the guard IS the stopping rule" begin
            # Stopping exactly on the boundary is not enough — completeness needs the pair beyond.
            @test !LZ.is_satisfied(ToValue(0.0), p[[1, 2, 3]],    Float64[], true)
            @test  LZ.is_satisfied(ToValue(0.0), p[[1, 2, 3, 4]], Float64[], true)
            @test !LZ.is_satisfied(ToValue(0.0), p[[5, 4, 3]],    Float64[], false)
            @test  LZ.is_satisfied(ToValue(0.0), p[[5, 4, 3, 2]], Float64[], false)
        end

        @testset "select drops the guard again" begin
            run = [1, 2, 3, 4]
            @test LZ.select(ToValue(0.0), run, p[run], Float64[], true) == [1, 2, 3]
        end
    end

    # ── Window ────────────────────────────────────────────────────────────────
    # The only spec whose rule lives in E, and the only one with NO p-space threshold:
    # its pool is the whole spectrum, so the run is limited only by where convergence stops.
    @testset "Window" begin
        @test LZ.needs_mapped(Window(0.5, 0.6))

        @testset "pool is the whole spectrum in walk order" begin
            @test LZ.pool_indices(Window(0.5, 0.6), p, false) == [5, 4, 3, 2, 1]
            @test LZ.pool_indices(Window(0.5, 0.6), p, true)  == [1, 2, 3, 4, 5]
        end

        @testset "straddle — both bounds must be bracketed" begin
            w = Window(0.5, 0.6)
            # E values of a contiguous run, in walk order.
            @test !LZ.is_satisfied(w, Float64[], [0.52, 0.58], false)          # in-window only
            @test !LZ.is_satisfied(w, Float64[], [0.52, 0.58, 0.63], false)    # above only
            @test !LZ.is_satisfied(w, Float64[], [0.45, 0.52, 0.58], false)    # below only
            @test  LZ.is_satisfied(w, Float64[], [0.52, 0.58, 0.63, 0.45], false)
            # Bounds are inclusive: a pair sitting ON a bound is in-window, not a bracket.
            @test !LZ.is_satisfied(w, Float64[], [0.5, 0.55, 0.6], false)
        end

        @testset "an infinite bound waives that side's bracket" begin
            # An edge window runs past the end of the spectrum: nothing exists there to
            # bracket, so waiting for one would hang until the budget ran out.
            wlo = Window(-Inf, 0.6)
            @test  LZ.is_satisfied(wlo, Float64[], [0.52, 0.63], false)   # only the above bracket
            @test !LZ.is_satisfied(wlo, Float64[], [0.52, 0.58], false)   # finite bound still demands its own

            whi = Window(0.45, Inf)
            @test  LZ.is_satisfied(whi, Float64[], [0.52, 0.40], false)
            @test !LZ.is_satisfied(whi, Float64[], [0.52, 0.58], false)

            # Both infinite = the whole spectrum, nothing to bracket at all.
            @test LZ.is_satisfied(Window(-Inf, Inf), Float64[], [0.5], false)
        end

        @testset "select keeps the in-window pairs + the NEAREST bracket per side" begin
            w   = Window(0.5, 0.6)
            run = [5, 4, 3, 2, 1]                       # walk order
            E   = [0.52, 0.58, 0.63, 0.45, 0.90]        # aligned with `run`
            # in-window {5,4}; nearest below 0.5 is 0.45 (index 2), NOT 0.90's far side;
            # nearest above 0.6 is 0.63 (index 3).
            @test sort(LZ.select(w, run, Float64[], E, false)) == [2, 3, 4, 5]

            # An infinite bound contributes no bracket: in-window is now {5,4,2} (E ≤ 0.6),
            # the high bracket is still the nearest above (index 3, E=0.63) — index 1 (E=0.90)
            # is above the bound too but is NOT the nearest, so it stays out.
            @test sort(LZ.select(Window(-Inf, 0.6), run, Float64[], E, false)) == [2, 3, 4, 5]
        end
    end
end
