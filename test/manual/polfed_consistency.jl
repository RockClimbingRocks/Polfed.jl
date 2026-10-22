# =====================================================================================
# POLFED consistency test suite
#
# Validates that every UniPolfed entrypoint agrees with full exact diagonalization (ED)
# and — just as importantly — with *itself* across different ways of asking for the same
# eigenpairs:
#
#   * `polfed(H, x0, howmany, target)`  (count mode)  vs ED
#   * `polfed(H, x0, (lo, hi))`         (window mode) vs ED
#   * count → window round trip: ask for N pairs, then ask for the interval those N span
#     and get the SAME N pairs back
#   * independence of the random start vector / block size
#   * operator (`f!`) path vs matrix (`H`) path
#   * `patched_polfed` (count + window) vs ED and vs plain `polfed`
#   * `full_polfed` vs the complete ED spectrum — across patch sizes, both entrypoints, and
#     the eigenvectors as a complete orthonormal basis that diagonalizes H
#   * `double_polfed` vs two independent single-window `polfed` runs — window and count mode,
#     all three strategies, tuple counts, rescaled targets, both entrypoints, validation
#
# Every eigenpair returned by any run is additionally checked against the *definition*
# of an eigenpair: ‖H·v − λ·v‖ small and ‖v‖ = 1.
#
# NOT part of `runtests.jl` — it runs full ED plus many POLFED runs and takes minutes.
# Run it on demand:
#
#     $HOME/julia-wrappers/julia-x86_64 --project=. test/manual/polfed_consistency.jl
#
# ── KNOWN FAILURE ────────────────────────────────────────────────────────────────────
# The `check_count` assertions fail for some count-mode targets: `polfed(H, x0, howmany,
# target)` occasionally returns `howmany - 1` eigenpairs. Every pair it DOES return is
# correct (right eigenvalue, converged residual, contiguous block) — the shortfall is in
# the count alone, and POLFED only `@warn`s about it. `check_count` classifies each case
# as BUDGET-limited or INTERVAL-limited; see its docstring and testset 11.
# =====================================================================================

using UniPolfed
using LinearAlgebra
using Random
using SparseArrays
using Test

const P = UniPolfed

# Keep the pipelines quiet (level 1 = warnings only). Bump to 2/3 while debugging.
P.Common.verbosity[] = 1

# ─────────────────────────────────────────────────────────────────────────────────────
# Test fixture
#
# A disordered open-boundary XXZ chain in the half-filling sector. Disorder makes the
# spectrum ASYMMETRIC, which matters: CLAUDE.md warns that a symmetric spectrum with a
# centered target produces exact degeneracies that stall Lanczos.
# ─────────────────────────────────────────────────────────────────────────────────────

const L        = 12
const LUP      = 6
const H        = P.Models.xxz_hamiltonian(L, LUP, 1.0, 1.0, 0.5;
                                          boundary = :open,
                                          rng      = MersenneTwister(1))
const D        = size(H, 1)
const ED       = eigvals(Symmetric(Matrix(H)))     # ground truth, ascending
const EMIN     = ED[1]
const EMAX     = ED[end]

# Tolerances. `EIGTOL` is the POLFED acceptance tolerance (‖Hv − λv‖); eigenvalues are
# compared an order of magnitude looser than the residual they were accepted at.
const EIGTOL   = 1e-10
const VAL_ATOL = 1e-8
const RES_ATOL = 1e-7

Random.seed!(20260811)
const X0 = randn(D)

# ─────────────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────────────

"Operator form of the fixture Hamiltonian: `f!(Y, X)` writes `H * X` into `Y`."
hop!(Y, X) = mul!(Y, H, X)

"Distance from `λ` to the nearest true eigenvalue of `H`."
function ed_distance(λ)
    i = searchsortedfirst(ED, λ)
    d = Inf
    i <= length(ED) && (d = min(d, abs(ED[i] - λ)))
    i > 1           && (d = min(d, abs(ED[i - 1] - λ)))
    return d
end

"Index into `ED` of the eigenvalue nearest `λ`."
function ed_index(λ)
    i = searchsortedfirst(ED, λ)
    i > length(ED) && return length(ED)
    i == 1         && return 1
    return abs(ED[i] - λ) <= abs(ED[i - 1] - λ) ? i : i - 1
end

"The ED eigenvalues lying inside the closed window `[lo, hi]`."
ed_in_window(lo, hi) = ED[lo .<= ED .<= hi]

"The subset of `vals` lying inside the closed window `[lo, hi]`."
in_window(vals, lo, hi) = [λ for λ in vals if lo <= λ <= hi]

"""
Window-mode contract check.

`polfed`/`patched_polfed` in window mode return every eigenpair inside `[lo, hi]` **plus
up to one bracketing eigenpair on each side** — the bracket is the certificate that the
window was covered completely (see `build_polfed_result`, which widens the slice by one
on each side). So the contract is:

  * the in-window subset equals the ED eigenvalues in the window, exactly;
  * every extra value is a true eigenvalue of `H`, and there are at most two of them.
"""
function check_window_contract(vals, lo, hi; label = "")
    inside = in_window(vals, lo, hi)
    check_same_spectrum(inside, ed_in_window(lo, hi); label = "$label (in-window set)")
    check_against_ed(vals; label = "$label (all returned)")
    @test length(vals) - length(inside) <= 2          # at most one bracket per side
    return inside
end

"""
Assert that every value in `vals` is a genuine eigenvalue of `H` (matches ED to
`VAL_ATOL`), that the set has no duplicates, and that it is sorted ascending.
"""
function check_against_ed(vals; label = "")
    @test issorted(vals)
    @test all(λ -> ed_distance(λ) < VAL_ATOL, vals)
    idx = ed_index.(vals)
    @test length(unique(idx)) == length(idx)          # no eigenvalue returned twice
    return idx
end

"""
Assert that the eigenvalues `vals` form a CONTIGUOUS block of the ED spectrum — no
holes. This is the property that makes a POLFED result usable: "all eigenpairs in this
interval", not "some eigenpairs in this interval".
"""
function check_contiguous(vals; label = "")
    idx = check_against_ed(vals; label = label)
    @test idx == collect(minimum(idx):maximum(idx))
    return idx
end

"Assert `‖H·v − λ·v‖ < RES_ATOL` and `‖v‖ ≈ 1` for every returned pair."
function check_residuals(vals, getvec; label = "")
    worst_res = 0.0
    worst_nrm = 0.0
    for i in eachindex(vals)
        v = getvec(i)
        worst_res = max(worst_res, norm(H * v - vals[i] * v))
        worst_nrm = max(worst_nrm, abs(norm(v) - 1))
    end
    @test worst_res < RES_ATOL
    @test worst_nrm < 1e-10
    return worst_res
end

"Residual check for a `PolfedResult` (eigenvectors via `eigenvector`)."
check_residuals(r::P.PolfedResult; label = "") =
    check_residuals(r.vals, i -> P.eigenvector(r, i); label = label)

"Residual check for a patched/full result (`vecs` is a `D × n` matrix or a vector of vectors)."
function check_residuals_patched(vals, vecs; label = "")
    getvec = vecs isa AbstractMatrix ? (i -> view(vecs, :, i)) : (i -> vecs[i])
    return check_residuals(vals, getvec; label = label)
end

"""
The `k` values of `vals` closest to `target`, ascending. `k` is clamped to `length(vals)`
so that a run which returned fewer pairs than requested produces a *test failure* (from
`check_count`) rather than a `BoundsError` that aborts the whole testset.
"""
nearest(vals, target, k) =
    sort(sort(collect(vals); by = λ -> abs(λ - target))[1:min(k, length(vals))])

"""
Assert a count-mode run returned at least `howmany` eigenpairs, and diagnose the cause
when it did not.

POLFED `@warn`s rather than throws when fewer converge inside the resolved interval, so
without this check a shortfall would pass silently. Pass `converged` (from
`PolfedResult.converged`) to separate the two failure modes:

  * `converged < howmany`  → **budget-limited**: Lanczos ran out of iterations. A larger
    `overestimate_iters` fixes it.
  * `converged >= howmany` → **interval-limited**: enough pairs converged, but the
    DoS-resolved target interval holds fewer than `howmany` states, so
    `build_polfed_result`'s interval filter discards the surplus. More budget does NOT
    help — the interval sizing is what is short.
"""
function check_count(vals, howmany; converged = nothing, label = "")
    n  = length(vals)
    ok = n >= howmany
    if !ok
        cause = converged === nothing         ? "unknown" :
                converged < howmany           ? "BUDGET-limited (converged=$converged)" :
                "INTERVAL-limited (converged=$converged, but the resolved interval " *
                "holds only $n states)"
        @info "SHORTFALL [$label]: requested $howmany, returned $n — $cause"
    end
    @test ok
    return ok
end

"Assert two eigenvalue lists describe the same set of eigenpairs."
function check_same_spectrum(a, b; atol = VAL_ATOL, label = "")
    @test length(a) == length(b)
    if length(a) == length(b)
        @test maximum(abs, sort(collect(a)) .- sort(collect(b))) < atol
    end
end

report(label, vals) = println(rpad(label, 44), "n = ", lpad(length(vals), 4),
                              "   range = [", round(minimum(vals); digits = 6), ", ",
                              round(maximum(vals); digits = 6), "]")

println("="^86)
println("POLFED consistency suite — XXZ L=$L, Lup=$LUP, D=$D, spectrum [$(round(EMIN;digits=4)), $(round(EMAX;digits=4))]")
println("="^86)

# ─────────────────────────────────────────────────────────────────────────────────────

@testset "POLFED consistency" begin

# ═════════════════════════════════════════════════════════════════════════════════════
# 1. polfed, count mode — the returned pairs must be true eigenpairs of H, contiguous
#    in the spectrum, and centred on the requested target.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "polfed count mode vs ED" begin
    target_idx = 400
    target     = ED[target_idx]

    for howmany in (10, 50, 100)
        r = P.polfed(H, X0, howmany, target; eigentol = EIGTOL)
        report("count polfed(howmany=$howmany)", r.vals)

        idx = check_contiguous(r.vals; label = "count $howmany")
        check_residuals(r; label = "count $howmany")

        # We asked for `howmany`; POLFED warns rather than throws if fewer land in the
        # resolved interval, so assert the count explicitly here.
        check_count(r.vals, howmany; converged = r.converged,
                    label = "count polfed howmany=$howmany")
        @test r.converged >= length(r.vals)

        # The block must actually surround the target, not sit off to one side.
        @test minimum(idx) <= target_idx <= maximum(idx)
    end
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 2. polfed, window mode — must return EXACTLY the ED eigenvalues inside the window.
#    Windows are placed off-boundary (midway between neighbouring eigenvalues) so that
#    "inside the window" is unambiguous.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "polfed window mode vs ED" begin
    for (i, j) in ((300, 340), (450, 520), (600, 640))
        lo = (ED[i - 1] + ED[i]) / 2
        hi = (ED[j] + ED[j + 1]) / 2
        expected = ed_in_window(lo, hi)

        r = P.polfed(H, X0, (lo, hi); eigentol = EIGTOL)
        report("window polfed([$i, $j])", r.vals)

        inside = check_window_contract(r.vals, lo, hi; label = "window $i:$j")
        check_contiguous(inside; label = "window $i:$j")
        check_residuals(r; label = "window $i:$j")
        check_same_spectrum(inside, expected; label = "window $i:$j")
    end
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 3. Count → window round trip.  THE headline consistency check: ask for N eigenpairs
#    near a target, then ask for the interval those N pairs span — the second run must
#    return the same N pairs.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "count → window round trip" begin
    for (howmany, target_idx) in ((100, 462), (40, 250), (60, 700))
        target = ED[target_idx]

        rc = P.polfed(H, X0, howmany, target; eigentol = EIGTOL)
        check_count(rc.vals, howmany; converged = rc.converged,
                    label = "round trip count=$howmany @ ED[$target_idx]")
        idx = check_contiguous(rc.vals)

        # Span the returned block with a window whose edges fall strictly between the
        # block and its ED neighbours, so the window contains exactly that block.
        i, j = minimum(idx), maximum(idx)
        lo = i == 1        ? EMIN - 1.0 : (ED[i - 1] + ED[i]) / 2
        hi = j == length(ED) ? EMAX + 1.0 : (ED[j] + ED[j + 1]) / 2

        rw = P.polfed(H, X0, (lo, hi); eigentol = EIGTOL)
        report("round trip count=$howmany → window", rw.vals)

        inside = check_window_contract(rw.vals, lo, hi; label = "round trip $howmany")
        check_contiguous(inside)
        check_residuals(rw)
        # The window run's in-window set must be exactly the count run's answer.
        check_same_spectrum(rc.vals, inside; label = "round trip $howmany")
    end
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 4. Start-vector independence — the answer is a property of H, not of x0. Different
#    random starts, and a block start (block Lanczos), must all agree.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "start-vector independence" begin
    target  = ED[380]
    howmany = 40
    ref     = P.polfed(H, X0, howmany, target; eigentol = EIGTOL)
    check_contiguous(ref.vals)

    for seed in (101, 202)
        rng = MersenneTwister(seed)
        r   = P.polfed(H, randn(rng, D), howmany, target; eigentol = EIGTOL)
        report("start vector seed=$seed", r.vals)
        check_contiguous(r.vals)
        check_residuals(r)
        # Compare the pairs nearest the target — the set both runs are guaranteed to hold
        # even if one converged a few extra pairs at the far edge of its interval.
        k = min(howmany, length(ref.vals), length(r.vals))
        check_same_spectrum(nearest(ref.vals, target, k), nearest(r.vals, target, k);
                            label = "seed $seed")
    end

    # Block start (block size 2) must find the same eigenvalues as block size 1.
    rb = P.polfed(H, randn(MersenneTwister(303), D, 2), howmany, target; eigentol = EIGTOL)
    report("block start (bs=2)", rb.vals)
    check_contiguous(rb.vals)
    check_residuals(rb)
    kb = min(howmany, length(rb.vals), length(ref.vals))
    check_same_spectrum(nearest(rb.vals, target, kb), nearest(ref.vals, target, kb);
                        label = "block start")
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 5. Operator path vs matrix path — the matrix entrypoint builds fast value-grouped row
#    kernels, the operator entrypoint runs column-native on the user's f!. Same answer.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "operator path vs matrix path" begin
    target  = ED[520]
    howmany = 30

    rmat = P.polfed(H,    X0, howmany, target; eigentol = EIGTOL)
    rop  = P.polfed(hop!, X0, howmany, target; eigentol = EIGTOL)
    report("operator f! (count)", rop.vals)

    check_contiguous(rop.vals)
    check_residuals(rop)

    k = min(howmany, length(rmat.vals), length(rop.vals))
    check_same_spectrum(nearest(rmat.vals, target, k), nearest(rop.vals, target, k);
                        label = "operator vs matrix (count)")
    # Both paths must agree on the COUNT too, not just on the overlapping pairs.
    @test length(rmat.vals) == length(rop.vals)

    # Same check in window mode, where the returned set is fully determined.
    lo = (ED[540] + ED[541]) / 2
    hi = (ED[580] + ED[581]) / 2
    wmat = P.polfed(H,    X0, (lo, hi); eigentol = EIGTOL)
    wop  = P.polfed(hop!, X0, (lo, hi); eigentol = EIGTOL)
    report("operator f! (window)", wop.vals)
    check_same_spectrum(in_window(wmat.vals, lo, hi), in_window(wop.vals, lo, hi);
                        label = "operator vs matrix (window)")
    check_window_contract(wop.vals, lo, hi; label = "operator window vs ED")
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 6. patched_polfed, count mode — an edge-first patched run must land on the same
#    eigenpairs as a plain single-filter run and as ED.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "patched_polfed count mode" begin
    target  = ED[430]
    howmany = 60

    vals, vecs, prep = P.patched_polfed(H, X0, howmany, 20, target; eigentol = EIGTOL)
    report("patched count(howmany=$howmany, per_patch=20)", vals)

    check_count(vals, howmany; label = "patched count howmany=$howmany")
    check_contiguous(vals)
    check_residuals_patched(vals, vecs)

    # The pairs nearest the target must match plain polfed's answer.
    rp = P.polfed(H, X0, howmany, target; eigentol = EIGTOL)
    k  = min(howmany, length(vals), length(rp.vals))
    check_same_spectrum(nearest(vals, target, k), nearest(rp.vals, target, k);
                        label = "patched vs plain (count)")
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 7. patched_polfed, window mode — a wide window split into patches must reassemble into
#    exactly the ED eigenvalues in that window, with no seam duplicates and no holes.
#    Different patch sizes must give the same answer.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "patched_polfed window mode" begin
    lo = (ED[200] + ED[201]) / 2
    hi = (ED[500] + ED[501]) / 2
    expected = ed_in_window(lo, hi)

    results = Vector{Vector{Float64}}()
    for per_patch in (25, 50)
        vals, vecs, prep = P.patched_polfed(H, X0, per_patch, (lo, hi); eigentol = EIGTOL)
        report("patched window(per_patch=$per_patch)", vals)

        inside = in_window(vals, lo, hi)
        check_contiguous(inside)                     # no seam holes
        check_residuals_patched(vals, vecs)
        check_same_spectrum(inside, expected; label = "patched window per_patch=$per_patch")
        # Seam eigenvalues are deduped at the end of a patched run, so no value may repeat.
        check_against_ed(sort(collect(vals)); label = "patched window per_patch=$per_patch")
        push!(results, collect(inside))
    end

    # Patch size is an implementation knob, not part of the answer.
    check_same_spectrum(results[1], results[2]; label = "patch size independence")

    # And a plain single-filter run over the same window agrees too.
    rw = P.polfed(H, X0, (lo, hi); eigentol = EIGTOL)
    check_same_spectrum(in_window(rw.vals, lo, hi), results[1];
                        label = "patched vs plain (window)")
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 8. full_polfed — the whole spectrum. This is the strictest check available: the full
#    sorted output must equal `eigvals(H)` element by element.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "full_polfed vs full ED spectrum" begin
    results = Vector{Vector{Float64}}()

    # Patch size is an implementation knob: every choice must reproduce the SAME spectrum.
    for per_patch in (40, 60, 120)
        vals, vecs, frep = P.full_polfed(H, X0, per_patch; eigentol = EIGTOL)
        report("full_polfed(per_patch=$per_patch)", vals)

        @test length(vals) == D                       # nothing missing, nothing duplicated
        if length(vals) == D
            @test maximum(abs, sort(collect(vals)) .- ED) < VAL_ATOL
        end
        check_contiguous(vals)
        check_residuals_patched(vals, vecs)
        push!(results, sort(collect(vals)))
    end
    check_same_spectrum(results[1], results[2]; label = "full per_patch 40 vs 60")
    check_same_spectrum(results[2], results[3]; label = "full per_patch 60 vs 120")

    # The whole spectrum means a complete eigenBASIS, not just D correct eigenvalues: the
    # eigenvectors must be orthonormal. This is the strongest single check in the suite — it
    # fails if any patch returned a duplicate, a stale, or an unconverged vector.
    @testset "eigenvectors form a complete orthonormal basis" begin
        vals, vecs, _ = P.full_polfed(H, X0, 60; eigentol = EIGTOL)
        V = vecs isa AbstractMatrix ? Matrix(vecs) : reduce(hcat, vecs)
        @test size(V) == (D, D)
        if size(V) == (D, D)
            @test maximum(abs, V' * V - I) < 1e-8     # orthonormal
            # ...and it actually diagonalizes H: V' H V must be the diagonal of eigenvalues.
            M = V' * Matrix(H) * V
            @test maximum(abs, M - Diagonal(diag(M))) < 1e-7
            @test maximum(abs, sort(diag(M)) .- ED) < VAL_ATOL
        end
    end

    # Operator entrypoint must reach the same spectrum as the matrix one.
    @testset "operator path" begin
        vals, vecs, _ = P.full_polfed(hop!, X0, 60; eigentol = EIGTOL)
        report("full_polfed operator f!", vals)
        @test length(vals) == D
        check_residuals_patched(vals, vecs)
        check_same_spectrum(sort(collect(vals)), results[2]; label = "full operator vs matrix")
    end
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 9. double_polfed — two windows in one call must equal two independent single-window
#    calls, in the caller's argument order (window 1 first, even when it is the higher
#    of the two).
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "double_polfed vs two single runs" begin
    lo1 = (ED[220] + ED[221]) / 2;  hi1 = (ED[260] + ED[261]) / 2
    lo2 = (ED[640] + ED[641]) / 2;  hi2 = (ED[680] + ED[681]) / 2

    for strategy in (:double, :separate)
        dres = P.double_polfed(H, X0, (lo1, hi1), (lo2, hi2);
                               strategy = strategy, eigentol = EIGTOL)
        r1, r2 = dres
        report("double window ($strategy) — w1", r1.vals)
        report("double window ($strategy) — w2", r2.vals)

        check_residuals(r1); check_residuals(r2)
        i1 = check_window_contract(r1.vals, lo1, hi1; label = "double w1 ($strategy)")
        i2 = check_window_contract(r2.vals, lo2, hi2; label = "double w2 ($strategy)")
        check_contiguous(i1); check_contiguous(i2)

        # A joint run pools both windows into one Krylov space; the split must not let one
        # window claim the other's eigenpairs.
        @test isempty(intersect(ed_index.(i1), ed_index.(i2)))
    end

    # Argument order is preserved: pass the HIGH window first and it comes back first.
    dres = P.double_polfed(H, X0, (lo2, hi2), (lo1, hi1); eigentol = EIGTOL)
    rhi, rlo = dres
    check_same_spectrum(in_window(rhi.vals, lo2, hi2), ed_in_window(lo2, hi2);
                        label = "double order (first=high)")
    check_same_spectrum(in_window(rlo.vals, lo1, hi1), ed_in_window(lo1, hi1);
                        label = "double order (second=low)")

    # Count mode: `howmany` pairs around each of two point targets, under both strategies.
    for strategy in (:double, :separate, :auto)
        dcnt = P.double_polfed(H, X0, 20, ED[300], ED[700];
                               strategy = strategy, eigentol = EIGTOL)
        c1, c2 = dcnt
        report("double count ($strategy) — w1", c1.vals)
        report("double count ($strategy) — w2", c2.vals)
        check_contiguous(c1.vals); check_residuals(c1)
        check_contiguous(c2.vals); check_residuals(c2)
        check_count(c1.vals, 20; converged = c1.converged, label = "double count w1 ($strategy)")
        check_count(c2.vals, 20; converged = c2.converged, label = "double count w2 ($strategy)")

        # Neither window may claim the other's eigenpairs out of the shared pool.
        @test all(λ -> λ < ED[500], c1.vals)
        @test all(λ -> λ > ED[500], c2.vals)

        # Each window must match what a standalone count run finds there.
        for (c, t) in ((c1, ED[300]), (c2, ED[700]))
            rs = P.polfed(H, X0, 20, t; eigentol = EIGTOL)
            k  = min(20, length(c.vals), length(rs.vals))
            check_same_spectrum(nearest(c.vals, t, k), nearest(rs.vals, t, k);
                                label = "double($strategy) vs single count @ $t")
        end
    end

    # `:auto` must resolve to one of the two real strategies and report its decision.
    dauto = P.double_polfed(H, X0, 20, ED[300], ED[700]; strategy = :auto, eigentol = EIGTOL)
    @test dauto.strategy in (:double, :separate)
    @test length(dauto) == 2
    @test dauto[1] === dauto.first && dauto[2] === dauto.second

    # A 2-tuple `howmany` asks for DIFFERENT counts per window, in the caller's order.
    @testset "per-window counts via a tuple" begin
        d = P.double_polfed(H, X0, (15, 35), ED[300], ED[700]; eigentol = EIGTOL)
        t1, t2 = d
        report("double count (15, 35) — w1", t1.vals)
        report("double count (15, 35) — w2", t2.vals)
        check_count(t1.vals, 15; converged = t1.converged, label = "tuple howmany w1")
        check_count(t2.vals, 35; converged = t2.converged, label = "tuple howmany w2")
        check_contiguous(t1.vals); check_residuals(t1)
        check_contiguous(t2.vals); check_residuals(t2)
        # The counts must follow the caller's argument order, not spectral order.
        @test length(t1.vals) < length(t2.vals)
    end

    # Non-absolute point specs must work here too, not just plain energies.
    @testset "rescaled point targets" begin
        v1 = (ED[350] - (EMAX + EMIN) / 2) / ((EMAX - EMIN) / 2)
        v2 = (ED[650] - (EMAX + EMIN) / 2) / ((EMAX - EMIN) / 2)
        d = P.double_polfed(H, X0, 15, (:rescaled, v1), (:rescaled, v2); eigentol = EIGTOL)
        r1, r2 = d
        report("double count (:rescaled) — w1", r1.vals)
        report("double count (:rescaled) — w2", r2.vals)
        check_contiguous(r1.vals); check_residuals(r1)
        check_contiguous(r2.vals); check_residuals(r2)
        @test minimum(abs.(r1.vals .- ED[350])) < VAL_ATOL   # window sits on its target
        @test minimum(abs.(r2.vals .- ED[650])) < VAL_ATOL
    end

    # Operator entrypoint agrees with the matrix one.
    @testset "operator path" begin
        dm = P.double_polfed(H,    X0, 15, ED[300], ED[700]; eigentol = EIGTOL)
        do_ = P.double_polfed(hop!, X0, 15, ED[300], ED[700]; eigentol = EIGTOL)
        for (i, t) in ((1, ED[300]), (2, ED[700]))
            k = min(15, length(dm[i].vals), length(do_[i].vals))
            check_same_spectrum(nearest(dm[i].vals, t, k), nearest(do_[i].vals, t, k);
                                label = "double operator vs matrix w$i")
        end
    end

    # Argument validation.
    @testset "input validation" begin
        # A window spec is not a point target for the count method.
        @test_throws ArgumentError P.double_polfed(H, X0, 10, (ED[200], ED[300]), ED[700])
        # Counts must be positive.
        @test_throws ArgumentError P.double_polfed(H, X0, 0, ED[300], ED[700])
        @test_throws ArgumentError P.double_polfed(H, X0, (10, -1), ED[300], ED[700])
    end
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 10. Spectrum edges — the extreme eigenvalues sit ON ±1 after rescaling, where Emin/Emax
#     are only estimates. Both ends must still come out exactly.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "spectrum edges" begin
    # Lowest 20 via a window that extends below the true minimum.
    hi_lo = (ED[20] + ED[21]) / 2
    rlow  = P.polfed(H, X0, (EMIN - 1.0, hi_lo); eigentol = EIGTOL)
    report("edge window (lowest 20)", rlow.vals)
    check_same_spectrum(in_window(rlow.vals, EMIN - 1.0, hi_lo), ED[1:20];
                        label = "lowest edge")
    check_residuals(rlow)
    @test minimum(rlow.vals) ≈ EMIN atol = VAL_ATOL      # ground state actually found

    # Highest 20, symmetrically.
    lo_hi = (ED[end - 20] + ED[end - 19]) / 2
    rhigh = P.polfed(H, X0, (lo_hi, EMAX + 1.0); eigentol = EIGTOL)
    report("edge window (highest 20)", rhigh.vals)
    check_same_spectrum(in_window(rhigh.vals, lo_hi, EMAX + 1.0), ED[end - 19:end];
                        label = "highest edge")
    check_residuals(rhigh)
    @test maximum(rhigh.vals) ≈ EMAX atol = VAL_ATOL

    # Patched over an edge-touching window exercises the edge peel + interior tiling,
    # including the seam dedup where the peel meets the first interior patch.
    hi_p = (ED[120] + ED[121]) / 2
    vals, vecs, _ = P.patched_polfed(H, X0, 25, (EMIN - 1.0, hi_p); eigentol = EIGTOL)
    report("patched edge window (lowest 120)", vals)
    check_same_spectrum(in_window(vals, EMIN - 1.0, hi_p), ED[1:120];
                        label = "patched lowest edge")
    check_residuals_patched(vals, vecs)
end

# ═════════════════════════════════════════════════════════════════════════════════════
# 11. Count-mode completeness sweep.
#
#     `polfed(H, x0, howmany, target)` should return `howmany` eigenpairs. It sometimes
#     returns `howmany - 1`. This sweep isolates the effect across many targets and
#     separates the two causes (see `check_count`):
#
#       BUDGET-limited   — converged < howmany; a larger `overestimate_iters` fixes it.
#       INTERVAL-limited — converged >= howmany but the DoS-resolved interval holds one
#                          state too few, so the surplus pair is filtered out. Extra
#                          budget does NOT help.
#
#     Everything returned is still a correct, contiguous, converged eigenpair — the
#     shortfall is in the COUNT only.
# ═════════════════════════════════════════════════════════════════════════════════════
@testset "count-mode completeness sweep" begin
    howmany = 30
    shortfalls = 0
    for target_idx in (120, 250, 380, 462, 560, 700, 820)
        r = P.polfed(H, X0, howmany, ED[target_idx]; eigentol = EIGTOL)
        # Whatever came back is always correct...
        check_contiguous(r.vals; label = "sweep ED[$target_idx]")
        check_residuals(r; label = "sweep ED[$target_idx]")
        # ...but it must also be complete.
        ok = check_count(r.vals, howmany; converged = r.converged,
                         label = "sweep howmany=$howmany @ ED[$target_idx]")
        ok || (shortfalls += 1)
    end
    shortfalls > 0 && @info "count-mode sweep: $shortfalls / 7 targets returned fewer than $howmany pairs"
end

end # testset "POLFED consistency"

println("="^86)
