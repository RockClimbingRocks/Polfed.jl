# The walker — where the module's invariant is enforced:
#
#   everything returned is a CONTIGUOUS run inward from the targeted extreme, in which
#   EVERY pair passed BOTH the Ritz and the eigen tolerance.
#
# These tests are the regression suite for gapped spectra. The last testset reproduces the
# concrete failure that motivated the redesign: a converged pair far outside a window,
# with unconverged pairs between it and the window, must not certify that window's edge.

using Test
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

# A filtered spectrum: `E` rescaled energies, `p` the filter applied to them (a unimodal
# peak at `center`, as a delta filter is). Returned ascending in `p`, as Lanczos hands
# back its Ritz values.
function _filtered_spectrum(; center=-0.220, width=0.010, lo=-0.30, hi=-0.15, n=40)
    Eg = collect(range(lo, hi, length=n))
    P(x) = 1 / (1 + ((x - center) / width)^2)
    pg = P.(Eg)
    ord = sortperm(pg)
    return Eg[ord], pg[ord], P
end

# Drive `decide` with explicit converged sets (Ritz and eigen can differ, to prove both gate).
function _decide(target, E, p; ritz_conv, eigen_conv=ritz_conv, force=false)
    ritz = [i in ritz_conv ? 0.0 : 1.0 for i in eachindex(p)]
    eigen_of = idxs -> (E[idxs], k -> idxs[k] in eigen_conv,
                        [idxs[k] in eigen_conv ? 0.0 : 1.0 for k in eachindex(idxs)])
    return LZ.decide(target, p, i -> ritz[i] < 0.5, ritz, eigen_of; force=force)
end

@testset "converged_run" begin

    @testset "converged_prefix — a hole is never stepped over" begin
        pool = [5, 4, 3, 2, 1]
        @test LZ.converged_prefix(pool, _ -> true)  == [5, 4, 3, 2, 1]
        @test LZ.converged_prefix(pool, _ -> false) == Int[]
        @test LZ.converged_prefix(pool, i -> i != 3) == [5, 4]      # truncates AT the hole
        # Pairs beyond the hole are converged but unreachable — that is the whole point.
        @test LZ.converged_prefix(pool, i -> i in (5, 4, 2, 1)) == [5, 4]
        @test LZ.converged_prefix(Int[], _ -> true) == Int[]
    end

    E, p, P = _filtered_spectrum()
    n = length(p)
    top(k) = Set(n-k+1:n)                 # the k most extremal (largest p) indices
    walk = collect(n:-1:1)                # high-end walk order

    @testset "NumEigs — count reached, and a hole blocking it" begin
        v = _decide(HighestEigs(5), E, p; ritz_conv = top(6))
        @test v.done && length(v.keep) == 5

        # Knock out the 2nd-most-extremal: the run truncates at the hole, nothing certified.
        knocked = setdiff(top(6), Set([n - 1]))
        @test !_decide(HighestEigs(5), E, p; ritz_conv = knocked).done
        # Forced (maxiter/breakdown): the partial run IS extracted — one pair, no gap.
        vf = _decide(HighestEigs(5), E, p; ritz_conv = knocked, force = true)
        @test !vf.done && vf.keep == [n]
    end

    @testset "the p-only pre-gate short-circuits an unsatisfiable check" begin
        # `NumEigs` reads only `p`, so a rule that already fails on the Ritz prefix cannot
        # start passing once the run shrinks: skip the eigenvector reconstruction entirely
        # and report nothing. `force` is what makes the partial run available. (A `Window`
        # has no such shortcut — its rule is in `E` — so it always reconstructs.)
        v = _decide(HighestEigs(5), E, p; ritz_conv = top(2))
        @test !v.done && isempty(v.keep)
        @test length(_decide(HighestEigs(5), E, p; ritz_conv = top(2), force = true).keep) == 2
    end

    @testset "BOTH gates are required — Ritz alone never certifies" begin
        # Ritz-converged all the way, but one pair fails the eigen tolerance: the run must
        # truncate there exactly as if it had failed the Ritz gate.
        v = _decide(HighestEigs(5), E, p;
                    ritz_conv = top(6), eigen_conv = setdiff(top(6), Set([n - 2])),
                    force = true)
        @test !v.done
        @test length(v.keep) == 2                     # stops at the eigen failure
    end

    @testset "worst_residual is the WORST in the pool (what the scheduler reads)" begin
        v = _decide(HighestEigs(5), E, p; ritz_conv = top(2))
        @test v.worst_residual == 1.0                 # unconverged pairs still in the pool
        v2 = _decide(HighestEigs(5), E, p; ritz_conv = top(5))
        @test v2.worst_residual == 0.0
    end

    # ── Window ────────────────────────────────────────────────────────────────
    w_lo, w_hi = -0.2320, -0.2100
    win  = Highest(Window(w_lo, w_hi))
    straddle = findfirst(k -> any(E[walk[1:k]] .< w_lo) && any(E[walk[1:k]] .> w_hi),
                         eachindex(walk))

    @testset "Window — straddling both bounds certifies, and the set has NO GAPS" begin
        v = _decide(win, E, p; ritz_conv = Set(walk[1:straddle]))
        @test v.done

        kept = sort(E[v.keep])
        spacing = minimum(diff(sort(E)))
        @test maximum(diff(kept)) ≈ spacing rtol=1e-8      # contiguous on the grid: no holes

        # Every in-window eigenvalue is present — the property the criterion exists to guarantee.
        inwin = filter(x -> w_lo <= x <= w_hi, E)
        @test all(x -> x in kept, inwin)

        # …enclosed by exactly one bracket on each side, each the NEAREST neighbour.
        @test length(kept) == length(inwin) + 2
        @test kept[1]   ≈ maximum(E[E .< w_lo])
        @test kept[end] ≈ minimum(E[E .> w_hi])
    end

    @testset "Window — in-window pairs alone do not certify (no brackets yet)" begin
        inwin = [i for i in eachindex(E) if w_lo <= E[i] <= w_hi]
        @test !_decide(win, E, p; ritz_conv = Set(inwin)).done
    end

    @testset "Window — a hole inside the window truncates the run" begin
        holed = setdiff(Set(walk[1:straddle]), Set([walk[3]]))
        @test !_decide(win, E, p; ritz_conv = holed).done
        # Even forced, only the pairs before the hole come back — never the ones past it.
        vf = _decide(win, E, p; ritz_conv = holed, force = true)
        @test !vf.done && length(vf.keep) == 2
    end

    @testset "Window — an edge window certifies sooner (one bracket waived)" begin
        only_hi = findfirst(k -> any(E[walk[1:k]] .> w_hi), eachindex(walk))
        v = _decide(Highest(Window(-Inf, w_hi)), E, p;
                    ritz_conv = Set(walk[1:only_hi]))
        @test v.done
        @test only_hi < straddle                      # strictly less work than a two-sided window
    end

    @testset "Window — transformed-boundary selection supports overlapping patch seams" begin
        p_left = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
        E_left = [0.9, 0.8, 0.1, 0.3, 0.5, 0.6]
        left = LZ.decide(
            Highest(Window(0.2, 0.7, 0.5)), p_left,
            _ -> true, zeros(6),
            idxs -> (E_left[idxs], _ -> true, zeros(length(idxs))),
        )

        p_right = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
        E_right = [0.1, 0.3, 0.9, 0.5, 0.6, 0.8]
        right = LZ.decide(
            Highest(Window(0.45, 0.85, 0.5)), p_right,
            _ -> true, zeros(6),
            idxs -> (E_right[idxs], _ -> true, zeros(length(idxs))),
        )

        left_vals = sort(E_left[left.keep])
        right_vals = sort(E_right[right.keep])
        @test left.done && right.done
        @test left_vals == [0.3, 0.5, 0.6]
        @test right_vals == [0.5, 0.6, 0.8]
        @test left_vals[2:end] == right_vals[1:2]
    end

    @testset "Window walker does not write debug output" begin
        debug_p = [0.1, 0.4, 0.6, 0.9]
        debug_E = [0.9, 0.7, 0.5, 0.3]
        mktemp() do _, output
            redirect_stdout(output) do
                LZ.decide(
                    Highest(Window(0.2, 0.8)), debug_p, _ -> true, zeros(4),
                    idxs -> (debug_E[idxs], _ -> true, zeros(length(idxs))),
                )
            end
            flush(output)
            seekstart(output)
            @test isempty(read(output, String))
        end
    end

    @testset "REGRESSION — a distant converged pair must not bracket a window" begin
        # The reported failure: obtained values [-0.25203, -0.22856, …] for the window
        # [-0.23102, -0.21023]. The old rule accepted ANY converged pair on the correct side
        # at ANY distance as a bracket, so a pair a full window-width below the low bound
        # certified the low edge while everything between was still unconverged.
        far_below = findfirst(x -> x < w_lo - 0.015, E)
        conv = union(Set(walk[1:max(1, straddle - 6)]), Set([far_below]))
        @test !_decide(win, E, p; ritz_conv = conv).done            # NOT certified

        # And it is never handed back as the low bracket: the walk cannot reach it past the
        # hole, so it is not in the run that `select` picks brackets from.
        @test !(far_below in _decide(win, E, p; ritz_conv = conv, force = true).keep)
    end

    @testset "BothEnds — union of two runs; the untouched middle is not a hole" begin
        v = _decide(BothEnds(3, 3), E, p; ritz_conv = union(top(3), Set(1:3)))
        @test v.done
        @test length(v.keep) == 6
        @test sort(v.keep) == sort(vcat(collect(1:3), collect(n-2:n)))

        # One end short → the whole target is unsatisfied.
        @test !_decide(BothEnds(3, 3), E, p; ritz_conv = top(3)).done
    end
end
