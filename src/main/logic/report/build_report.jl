# Post-processing: turn the raw Lanczos result into the returned PolfedResult. Three steps
# fold in here (the correctness "notes to myself" from main.jl):
#   1. filter to the target interval — keep only eigenvalues inside the energy window the
#      filter exposed (mapped back from the rescaled [left, right]);
#   2. sort ascending by true energy of H (Lanczos already returns sorted, and the in-interval
#      subset of a sorted vector is a contiguous range, so slicing stays a view);
#   3. warn (never throw) if fewer than `howmany` eigenpairs landed inside the interval.
# It then assembles the Mapping/Transform snapshots and the full Report.

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
    build_polfed_result(lr::LanczosResult, mp::MappingPlan, interval::Interval,
                        spec::TargetSpec, filter::AbstractFilter, layout::Layout,
                        howmany::Int, timer::StageTimer; vals_rescaled::Bool=false) -> PolfedResult

Post-process the raw [`LanczosResult`](@ref) into the returned [`PolfedResult`](@ref):

1. Un-rescale `lr.vals` and `lr.residuals` to true-energy units when `vals_rescaled` (see below).
2. Compute the true-energy target window from `interval` and the mapping.
3. Keep the eigenvalues inside it. Lanczos returns `vals` sorted ascending, so the in-interval
   set is the contiguous range `searchsortedfirst … searchsortedlast` — the eigenvector subset is
   taken as a **view** (columns for [`ColumnLayout`](@ref), rows for [`RowLayout`](@ref)), no copy.
   The trim is safe precisely because the Lanczos criterion guarantees a **contiguous, gap-free**
   converged run: slicing it can only drop pairs from the ends, never punch a hole. For a window
   target the slice is **widened by one on each side**, keeping the bracketing eigenpairs the
   criterion selected (see [`Window`](@ref)) — a window result encloses its interval.
4. `@warn` (not error) if fewer than `howmany` eigenpairs fall inside the interval, or if the
   criterion could not certify the window complete.

# Keywords
- `vals_rescaled`: `true` when the pipeline ran Lanczos against a **rescaled** `mapvals`, so
  `lr.vals` are energies of `H̃` rather than of `H` (the window pipeline — its [`Window`](@ref)
  criterion is stated in rescaled coordinates). Both values and residuals are mapped back here,
  the single un-rescaling point: `E = mp.a·Ẽ + mp.b` and `‖H·v − λ·v‖ = mp.a·‖H̃·v − λ̃·v‖`. The
  pipeline correspondingly passed `eigentol / mp.a` to Lanczos, so the acceptance test means
  `‖H·v − λ·v‖ < eigentol` exactly as it does for a count target. (The [`FactorizationReport`](@ref)
  keeps Lanczos' own rescaled numbers — tolerance and residual there are consistent with each
  other, just a factor `mp.a` smaller than the true-energy figures reported here.)

Then snapshot the [`MappingReport`](@ref)/[`SpectralTransformReport`](@ref), assemble the
[`Report`](@ref) (carrying `timer`, the pipeline's POLFED-level [`StageTimer`](@ref)), and return
the `PolfedResult`.
"""
function build_polfed_result(lr::LanczosResult, mp::MappingPlan, interval::Interval,
                             spec::TargetSpec, filter::AbstractFilter, layout::Layout,
                             howmany::Int, timer::StageTimer; vals_rescaled::Bool=false)
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
    lo = searchsortedfirst(valscpu, Elo)
    hi = searchsortedlast(valscpu, Ehi)
    n_in = max(0, hi - lo + 1)

    # Window mode returns the interval BRACKETED: one eigenpair below `low` and one above
    # `high`, so the caller can see the interval is enclosed (and so adjacent patches can be
    # glued at a verified seam). Widen the slice by one on each side to keep them; the
    # criterion already selected exactly those two, and a missing one (edge window, or a
    # bracket that never converged) just means there is nothing there to widen onto.
    if spec isa TargetWindowSpec
        lo = max(1, lo - 1)
        hi = min(length(valscpu), hi + 1)
    end

    kept_vals = valscpu[lo:hi]
    kept_res  = residuals[lo:hi]
    kept_vecs = layout isa RowLayout ? view(lr.vecs, lo:hi, :) : view(lr.vecs, :, lo:hi)

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
        polfed_log(POLFED_WARN_LEVEL,
                   "POLFED: fewer eigenpairs converged inside the target interval than requested. " *
                   "Returning the ones found.",
                   requested = howmany, in_interval = n_in, converged = fr.converged_tol)
    end

    mapping_report   = MappingReport(mp, layout)
    transform_report = SpectralTransformReport(interval, filter, mp, spec,
                                               fr.blocksize, fr.itersneeded, howmany)
    report = Report(mapping_report, transform_report, fr, timer)

    return PolfedResult(kept_vals, kept_vecs, collect(Float64, kept_res), lr.converged, layout, report)
end
