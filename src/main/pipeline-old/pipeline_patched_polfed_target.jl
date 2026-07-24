"""
    run_patched_polfed_target_pipeline(f!, x0, howmany, howmany_per_patch, target; kwargs...) -> (vals, vecs)

Count/target pipeline for [`patched_polfed`](@ref): return the `howmany` eigenpairs nearest a
target energy by covering the window that holds them with a sequence of narrower **patches**,
running one polynomial-filtered Lanczos per patch, and gluing the per-patch spectra into one sorted
set. It is the patched counterpart of [`run_polfed_target_pipeline`](@ref): a single delta filter's
order `K` blows up as the window widens, so splitting the window keeps each patch's `K` (and Krylov
space) small.

The only difference from [`run_patched_polfed_window_pipeline`](@ref) is where the window comes
from: here it is *derived* from `(target, howmany)` — [`resolve_target`](@ref) places the rescaled
center and [`resolve_interval`](@ref) grows it (via the DoS) into the window that holds `howmany`
states — rather than being given explicitly. Everything downstream (partition → per-patch
filter/transform/Lanczos → ownership glue) is identical.

# Pipeline
1. **Mapping** ([`build_mapping_plan`](@ref)): rescale `H → H̃ = (H-b)/a` into `[-1, 1]`.
2. **DoS** ([`calculate_dos`](@ref)) on the rescaled operator.
3. **Target → window** ([`resolve_target`](@ref) → [`resolve_interval`](@ref)): resolve the rescaled
   target center, then grow it into the window `full_iv` that holds `howmany` states.
4. **Partition** ([`intervalize_interval_overlapping`](@ref)): split `full_iv` into `unpadded`
   patches, each grown outward by `Defaults.patched_polfed.howmany_pad` states into an overlapping
   `padded` patch. Each patch is *converged* over its padded window but *owns* only its unpadded
   interval — the padding keeps every owned eigenvalue clear of the fragile bracket edges of its run.
5. **Per patch** — resolve a filter for the *padded* patch, transform, and run Lanczos with a
   [`Window`](@ref) target over the padded patch. **The filter type is chosen by geometry**
   ([`resolve_filter`](@ref)): interior patches take a delta filter, patches touching a spectral
   edge take an [`EdgeFilter`](@ref). Convergence adapts automatically — nothing here reads a
   delta-only field like `passband`; the per-patch Krylov budget comes from the DoS over the patch,
   and [`window_eig_target`](@ref) encodes an edge-touching bound as `±Inf` (no bracket to wait for
   on the spectrum-end side).
6. **Glue** ([`patch_results_owned!`](@ref)): keep each patch's eigenpairs that fall in its
   *unpadded* interval. Those intervals tile the window with shared exact endpoints, so half-open
   ownership assigns every eigenvalue to exactly one patch (no gaps, no duplicates) — robust to the
   `Window` brackets that make strict prefix-alignment misalign. Then un-rescale to true energies.

# Keywords
- `mc::MappingConfig`, `dosc::DoSConfig`: per-stage configuration.
- `tol`, `eigentol::Real`: Ritz / eigen residual tolerances (true-energy `eigentol`).
- `overestimate_iters::Real`: Krylov-budget safety factor (floored at `2.5`, as for the window
  pipeline — the near-edge brackets of a `Window` converge late).
- `layout::Layout`: Krylov block layout.

Returns `(vals, vecs)` in **true** energies, ascending: the `howmany` eigenpairs nearest the target
(fewer only if that many did not converge, with a warning). Unlike the window pipeline — where the
count is not known a priori, so eigenvectors accumulate in a vector-of-vectors — here `howmany` is
known, so `vecs` is a **dense matrix**: `dim × k` for a [`ColumnLayout`](@ref), `k × dim` for a
[`RowLayout`](@ref), where `k = min(howmany, converged)`. (Placeholder return shape; a
`PolfedResult` + `Report` will replace it once the patched path is finalized.)
"""
function run_patched_polfed_target_pipeline(f!, x0::AbstractVecOrMat, howmany::Integer,
                howmany_per_patch::Integer, target;
                mc::MappingConfig        = MappingConfig(),
                dosc::DoSConfig          = DoSConfig(),
                tol::Real                = Defaults.polfed.tol,
                eigentol::Real           = Defaults.polfed.eigentol,
                overestimate_iters::Real = Defaults.polfed.overestimate_iters,
                layout::Layout           = ColumnLayout(),
)
    timer = StageTimer()

    # 1. Mapping: rescale H → H̃ = (H - b)/a into [-1, 1] (estimating Emin/Emax if needed).
    mp::MappingPlan   = @stage timer :bounds build_mapping_plan(mc, f!, x0)
    # 2. DoS on the rescaled operator (layout-agnostic — moments are spectral).
    dos::DoSResult    = @stage timer :dos calculate_dos(dosc, mp.f!_rescaled, mp.sys)
    # 3. Target → window: resolve the rescaled target center, then grow it into the window that
    #    holds `howmany` states around it (the region the patches will cover).
    spec::TargetSpec        = normalize_target_spec(target)
    rescaled_target::Target = @stage timer :interval resolve_target(spec, howmany, dos, mp.a, mp.b)
    full_iv::Interval       = @stage timer :interval resolve_interval(rescaled_target, dos)

    polfed_log(POLFED_INFO_LEVEL, "Starting patched POLFED (count target).",
               target = target, howmany = Int(howmany),
               window = (full_iv.left, full_iv.right),
               howmany_per_patch = Int(howmany_per_patch), layout = typeof(layout))

    # 4. Converge + glue the whole window edge-first; result is in RESCALED coordinates, ascending.
    #    `records` is one PatchReport per filtered-Lanczos run (edge peels + interior patches).
    vals, vecs, records = _patched_glued_spectrum(mp, dos, full_iv, howmany_per_patch,
                                                  tol, eigentol, overestimate_iters, layout, timer)

    # Un-rescale the accumulated Ritz values from rescaled [-1, 1] back to true energies.
    @. vals = vals * mp.a + mp.b

    # Count target: keep exactly the `howmany` eigenpairs nearest the target energy (the window was
    # sized to hold ~howmany, so gluing may yield a few more or fewer). Selecting by distance keeps
    # a contiguous block around the target — dropping only the farthest extras at the window rim.
    target_energy = rescaled_target.rescaled_energy * mp.a + mp.b
    k = min(Int(howmany), length(vals))
    k < howmany && polfed_log(POLFED_WARN_LEVEL,
        "Patched POLFED converged fewer eigenpairs than requested near the target.",
        requested = Int(howmany), converged = length(vals))
    sel = sort!(partialsortperm(abs.(vals .- target_energy), 1:k))   # k nearest, back in ascending order
    vals_out = vals[sel]

    # `howmany` is known, so pack the selected eigenvectors into a dense matrix (one column/row per
    # pair, per `layout`) instead of the window pipeline's vector-of-vectors.
    dim = mp.sys.dim
    vecs_out = layout isa RowLayout ? Matrix{Float64}(undef, k, dim) : Matrix{Float64}(undef, dim, k)
    for (slot, idx) in enumerate(sel)
        if layout isa RowLayout
            @views vecs_out[slot, :] .= vecs[idx]
        else
            @views vecs_out[:, slot] .= vecs[idx]
        end
    end

    report = PatchedReport(MappingReport(mp, layout), full_iv, Int(howmany_per_patch),
                           Defaults.patched_polfed.howmany_edge, Defaults.patched_polfed.howmany_pad,
                           records, k, mp.a, mp.b, timer)

    polfed_log(POLFED_INFO_LEVEL, "Patched POLFED finished.",
               eigenpairs = k, runs = length(records), walltime_s = total_wall(timer))

    return vals_out, vecs_out, report
end
