# Post-processing: turn the raw Lanczos result into the returned PolfedResult. Three steps
# fold in here (the correctness "notes to myself" from main.jl):
#   1. select the returned subset — WHICH boundary applies depends on the mode (see below);
#   2. sort ascending by true energy of H (Lanczos already returns sorted, and every selection
#      below is a contiguous range of that sorted vector, so slicing stays a view);
#   3. warn (never throw) if fewer than `howmany` eigenpairs were selected.
# It then assembles the Mapping/Transform snapshots and the full Report.
#
# ── Two boundaries, two modes ────────────────────────────────────────────────────────────────
# `interval` and `passband(filter)` are different things and must not be conflated:
#
#   * `interval` is the REQUEST. In window mode it is literally what the user asked for. In count
#     mode it is an internal artifact — the DoS-bisected window sized to hold `howmany` states.
#   * `passband(filter)` is the TRUST boundary: where the filter polynomial is above cutoff and
#     Ritz ordering therefore faithfully tracks energy ordering. Outside it, Chebyshev tails can
#     lift a distant eigenvalue above a nearer one, so a pair out there is not certifiable.
#     It is generally WIDER than the interval (`order_safety_factor < 1` widens it deliberately).
#
# Window mode trims at the `interval`: that is the request, and the (wider) passband would hand
# back pairs nobody asked for. Count mode must NOT trim at the interval — the user asked for
# "`howmany` nearest the target", not "everything inside a DoS-estimated window". Trimming there
# made the returned COUNT a function of the DoS estimation error: when the stochastic KPM density
# came out a fraction of a state optimistic, the interval spanned `howmany - 1` true eigenvalues
# and an already-converged pair was discarded. So count mode gates at the passband (discard what
# the filter cannot certify) and then selects the `howmany` nearest the target.
#
# That selection is contiguous, and legitimately so: within a delta filter's passband the
# polynomial decreases monotonically with |E − center| on each side of the peak, so the converged
# set is a contiguous energy band around the target and "nearest the target" coincides with
# "contiguous run inward". That equivalence is exactly what breaks outside the passband — hence
# the gate has to come first.

"""
    _interval_energy_window(mp::MappingPlan, interval::Interval) -> (Float64, Float64)

The target window in **true energy**: the rescaled `[left, right] ⊆ [-1, 1]` mapped through the
rescaling `E = spread·x̃ + center` (`mp.a`/`mp.b`). Since `mp.a > 0`, order is preserved.

A bound sitting on a rescaled spectrum edge (`±1`, e.g. an edge-filter window clamped there)
means the window runs to the **end of the spectrum**, so that side is returned as `∓Inf` and the
trim keeps everything: the mapping's `Emin`/`Emax` estimates sit *on* the extremal eigenvalues
(no margin), so trimming at the finite image would drop the spectrum-edge eigenpair itself to
reconstruction rounding. Mirrors the `±Inf` encoding of [`window_eig_target`](@ref).
"""
_interval_energy_window(mp::MappingPlan, interval::Interval) =
    (interval.left  <= -1.0 + 1e-9 ? -Inf : mp.a * interval.left  + mp.b,
     interval.right >=  1.0 - 1e-9 ?  Inf : mp.a * interval.right + mp.b)

"""
    _nearest_contiguous_range(vals, glo, ghi, target, howmany) -> (lo, hi)

The `howmany` entries of the **sorted** `vals[glo:ghi]` closest to `target`, as the index range
`lo:hi`.

The result is contiguous because the `k` nearest entries to a scalar in a sorted vector always
are: two pointers start at the target's insertion point and each step extends whichever side is
closer, so the range only ever grows outward. Keeping it a range (rather than an index set) is
what lets the caller take the eigenvector subset as a **view**.

Returns the whole candidate range when `howmany` meets or exceeds its length, and the empty range
`glo:glo-1` when `ghi < glo`.
"""
function _nearest_contiguous_range(vals::AbstractVector, glo::Int, ghi::Int,
                                   target::Real, howmany::Int)
    ghi >= glo || return (glo, glo - 1)                 # nothing survived the gate
    howmany >= ghi - glo + 1 && return (glo, ghi)       # everything qualifies

    # `r` is the first candidate at or above the target; `l` the last one below it. The range
    # `l+1 : r-1` (initially empty) grows one element per step, always toward the nearer side.
    r = clamp(searchsortedfirst(vals, target), glo, ghi + 1)
    l = r - 1
    for _ in 1:howmany
        take_left = if l < glo
            false                                       # left exhausted
        elseif r > ghi
            true                                        # right exhausted
        else
            abs(vals[l] - target) <= abs(vals[r] - target)
        end
        take_left ? (l -= 1) : (r += 1)
    end
    return (l + 1, r - 1)
end

"""
    build_polfed_result(lr::LanczosResult, mp::MappingPlan, interval::Interval,
                        spec::TargetSpec, filter::AbstractFilter, layout::Layout,
                        howmany::Int, timer::StageTimer; vals_rescaled::Bool=false) -> PolfedResult

Post-process the raw [`LanczosResult`](@ref) into the returned [`PolfedResult`](@ref):

1. Un-rescale `lr.vals` and `lr.residuals` to true-energy units when `vals_rescaled` (see below).
2. Select the returned subset, by a rule that depends on the mode:
   - **window target** — trim to `interval` (the request), then widen by one on each side to keep
     the bracketing eigenpairs the criterion selected (see [`Window`](@ref)), so a window result
     encloses its interval. A bracket lies just *outside* the [`passband`](@ref) by construction,
     so only `bracket_window` limits it.
   - **count target** — gate at the [`passband`](@ref), then keep the `howmany` values nearest
     `target_energy` ([`_nearest_contiguous_range`](@ref)). The DoS-sized `interval` does **not**
     bound the answer here; see the discussion at the top of this file for why. On the
     single-filter path the criterion already hands back exactly `howmany`, so that selection is
     a no-op and the returned set is the `howmany` nearest the filter PEAK (which is `p` order,
     not energy order — `P` is symmetric about `0`, not about a non-central target). It bites
     only when the caller pools more than one window's worth into one `LanczosResult`, i.e. the
     joint double-delta run.
   Either rule yields a contiguous range of the ascending `lr.vals`, so the eigenvector subset is
   taken as a **view** (columns for [`ColumnLayout`](@ref), rows for [`RowLayout`](@ref)), no
   copy. Slicing is safe precisely because the Lanczos criterion guarantees a **contiguous,
   gap-free** converged run: it can only drop pairs from the ends, never punch a hole.
3. `@warn` (not error) if fewer than `howmany` eigenpairs were selected, or if the criterion could
   not certify the window complete.

# Keywords
- `vals_rescaled`: `true` when the pipeline ran Lanczos against a **rescaled** `mapvals`, so
  `lr.vals` are energies of `H̃` rather than of `H` (the window pipeline — its [`Window`](@ref)
  criterion is stated in rescaled coordinates). Both values and residuals are mapped back here,
  the single un-rescaling point: `E = mp.a·Ẽ + mp.b` and `‖H·v − λ·v‖ = mp.a·‖H̃·v − λ̃·v‖`. The
  pipeline correspondingly passed `eigentol / mp.a` to Lanczos, so the acceptance test means
  `‖H·v − λ·v‖ < eigentol` exactly as it does for a count target. (The [`FactorizationReport`](@ref)
  keeps Lanczos' own rescaled numbers — tolerance and residual there are consistent with each
  other, just a factor `mp.a` smaller than the true-energy figures reported here.)
- `bracket_window`: `(lo, hi)` bounds, in the SAME units as the returned values (true energy),
  outside which this result may not claim an eigenpair — a window-mode bracket, or a count-mode
  candidate. In count mode it is intersected with the filter's [`passband`](@ref); in window mode
  it stands alone (a bracket lives outside the passband by design). A caller that post-processes ONE pooled
  `LanczosResult` into several windows (the joint double-delta run, whose one passband spans BOTH
  peaks and so cannot separate them) passes the neighbouring window's near edge, so a window can
  never claim the neighbour's eigenpairs.
- `target_energy`: the count-mode target in true energy — the point the `howmany` returned pairs
  are nearest to. Required for the count-mode selection described above; when left `NaN` the
  count path falls back to the old interval trim, so a caller that cannot name its target still
  works (with the DoS sensitivity that entails). Ignored for window targets.

Then snapshot the [`MappingReport`](@ref)/[`SpectralTransformReport`](@ref), assemble the
[`Report`](@ref) (carrying `timer`, the pipeline's POLFED-level [`StageTimer`](@ref)), and return
the `PolfedResult`.
"""
function build_polfed_result(lr::LanczosResult, mp::MappingPlan, interval::Interval,
                             spec::TargetSpec, filter::AbstractFilter, layout::Layout,
                             howmany::Int, timer::StageTimer; vals_rescaled::Bool=false,
                             bracket_window::Tuple{Float64,Float64}=(-Inf, Inf),
                             target_energy::Float64=NaN)
    fr = lr.report
    Elo, Ehi = _interval_energy_window(mp, interval)

    # In-interval subset (contiguous, since vals is sorted ascending). Search on a CPU copy so
    # the GPU path stays correct; the eigenvector subset is a lazy view of the original storage.
    valscpu = lr.vals isa Vector ? lr.vals : collect(lr.vals)
    residuals = lr.residuals
    if vals_rescaled                                        # the one un-rescaling point
        valscpu   = mp.a .* valscpu .+ mp.b
        residuals = mp.a .* residuals                       # ‖H·v − λ·v‖ = a·‖H̃·v − λ̃·v‖
    end
    if spec isa TargetWindowSpec
        # ── Window mode: the interval IS the request, so trim at it. ──
        lo = searchsortedfirst(valscpu, Elo)
        hi = searchsortedlast(valscpu, Ehi)
        n_in = max(0, hi - lo + 1)

        # Return the interval BRACKETED: one eigenpair below `low` and one above `high`, so the
        # caller can see the interval is enclosed (and so adjacent patches can be glued at a
        # verified seam). Widen the slice by one on each side to keep them; a missing one (edge
        # window, or a bracket that never converged) just means there is nothing to widen onto.
        #
        # NOTE the passband is deliberately NOT a bound here, unlike the count branch below. A
        # bracket lies just OUTSIDE the passband by construction — that is what makes it a
        # certificate that the window's edge was passed — so gating brackets at the passband would
        # simply delete them. The price is that a bracket need not be the immediate spectral
        # neighbour: below the passband the filter's tail oscillates, so the converged run can
        # reach a more distant state before a nearer one (see `window_entrypoint_tests.jl`, which
        # measures exactly this and asserts only side and existence).
        #
        # `bracket_window` remains the one limit: a pooled run — the joint double-delta run, whose
        # single Krylov space holds BOTH windows — passes the valley midpoint, so a window whose
        # own bracket never converged cannot widen onto the OTHER window's extremal eigenpair.
        lo > 1               && valscpu[lo - 1] >= bracket_window[1] && (lo -= 1)
        hi < length(valscpu) && valscpu[hi + 1] <= bracket_window[2] && (hi += 1)
    elseif isnan(target_energy)
        # ── Count mode, no target supplied: fall back to the interval trim. ──
        # Kept so a caller that cannot name its target energy still behaves as before; it carries
        # the DoS-sensitivity described at the top of this file.
        lo = searchsortedfirst(valscpu, Elo)
        hi = searchsortedlast(valscpu, Ehi)
        n_in = max(0, hi - lo + 1)
    else
        # ── Count mode: gate at the trust boundary, then take the `howmany` nearest target. ──
        # The DoS-sized `interval` is deliberately NOT consulted: it did its job upstream (sizing
        # the filter order and the Krylov budget), and letting it bound the answer is what made
        # the returned count depend on the DoS estimation error.
        #
        # Here the passband IS the right bound (unlike the bracket widening above): a count target
        # returns a SET of certified pairs, not a certificate of enclosure, so a pair from the
        # oscillating tail — which may not even be the nearest unconverged state — must not be
        # among them. `bracket_window` tightens it further for a pooled run, whose single
        # double-peak passband cannot by itself keep one window out of the other's eigenpairs.
        Plo, Phi = _interval_energy_window(mp, passband(filter))
        glo = searchsortedfirst(valscpu, max(Plo, bracket_window[1]))
        ghi = searchsortedlast(valscpu,  min(Phi, bracket_window[2]))
        lo, hi = _nearest_contiguous_range(valscpu, glo, ghi, target_energy, howmany)
        n_in = max(0, hi - lo + 1)
    end

    # Which rule ran, and what it did to the pool, is the whole answer to "why did I get n and
    # not howmany" — and it is invisible in the returned object, which only shows the survivors.
    should_log(POLFED_DEBUG_LEVEL) && polfed_log(POLFED_DEBUG_LEVEL, "Result selection.",
        rule = spec isa TargetWindowSpec ? "window: trim at the request, widen by one bracket" :
               isnan(target_energy) ? "count: interval trim (no target energy given)" :
               "count: passband gate, then the howmany nearest the target",
        from_lanczos = length(valscpu), in_interval = n_in, returned = max(0, hi - lo + 1),
        requested = howmany,
        energies = hi >= lo ? _fmt_iv(valscpu[lo], valscpu[hi]) : "none",
        bracket_window = _fmt_iv(bracket_window[1], bracket_window[2]))

    kept_vals = valscpu[lo:hi]
    kept_res  = residuals[lo:hi]
    # The Krylov basis may be stored row-major (the matrix entrypoints work in RowLayout), but the
    # RETURNED eigenvectors are always columns: `vecs[:, i]` is the i-th eigenvector regardless of
    # how the run was laid out internally. Row layout therefore costs one transpose here — a real
    # copy, not a lazy adjoint, so the result is a plain contiguous matrix.
    kept_vecs = layout isa RowLayout ? permutedims(view(lr.vecs, lo:hi, :)) : view(lr.vecs, :, lo:hi)

    # Window mode: the criterion itself reports whether it certified the window complete (its
    # stopping rule was met) or merely handed back its contiguous run at the budget ceiling.
    # Either way the returned set has NO GAPS — an uncertified one is only possibly SHORT at
    # the edges. This is read off the report rather than re-derived from residuals here.
    if spec isa TargetWindowSpec && !fr.complete
        polfed_log(POLFED_WARN_LEVEL,
                   "POLFED window mode: the target interval could not be CERTIFIED complete — the " *
                   "Krylov budget ran out before the converged run bracketed both window edges, so " *
                   "in-window eigenpairs near an edge may be missing. The returned set is still " *
                   "gap-free. Increase `overestimate_iters`.",
                   maxdim = fr.itersreserved, reached = fr.itersneeded, in_interval = n_in)
    end

    # Count-based targets have a genuine "requested" count to fall short of. Window targets don't —
    # `howmany` there is only the DoS budget estimate, so the number found IS the answer, not a shortfall.
    if !(spec isa TargetWindowSpec) && n_in < howmany
        # Two different shortfalls wear the same face here, and the fix for one is useless for
        # the other. `converged_tol > converged_eigentol` means the Krylov space DID reach the
        # requested count and the eigen tolerance is what threw pairs away; widening the budget
        # then buys nothing but runtime.
        eigen_limited = fr.converged_tol > fr.converged
        advice = eigen_limited ?
            "The Krylov space reached the requested count — `eigentol` is what cut the run, so " *
            "a larger budget will not help. Relax `eigentol`, or narrow the window so the " *
            "filter separates neighbouring states more strongly." :
            "Increase `overestimate_iters` to widen the Krylov budget."
        polfed_log(POLFED_WARN_LEVEL,
                   "POLFED: fewer eigenpairs converged near the target than requested. " *
                   "Returning the ones found. " * advice,
                   requested = howmany, returned = n_in,
                   converged = fr.converged, passed_ritz = fr.converged_tol,
                   passed_eigen = fr.converged_eigentol, eigentol = fr.eigentol,
                   worst_eigen_residual = fr.eigenworst)
    end

    mapping_report   = MappingReport(mp, layout)
    transform_report = SpectralTransformReport(interval, filter, mp, spec,
                                               fr.blocksize, fr.itersneeded, howmany)
    report = Report(mapping_report, transform_report, fr, timer)

    # `layout` here describes the RETURNED eigenvectors, which are column-oriented either way.
    # The Krylov layout the run actually used is recorded in the mapping report.
    return PolfedResult(kept_vals, kept_vecs, collect(Float64, kept_res), lr.converged,
                        ColumnLayout(), report)
end
