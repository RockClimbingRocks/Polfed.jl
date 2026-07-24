"""
    run_patched_polfed_window_pipeline(f!, x0, howmany_per_patch, target; kwargs...) -> (vals, vecs)

Window pipeline for [`patched_polfed`](@ref): return **all** eigenpairs in a *wide* rescaled window
by covering it edge-first (see [`_patched_glued_spectrum`](@ref)) and gluing the pieces into one
sorted set. A single delta filter's order `K` blows up as the window widens, so the interior is
split into narrow patches (small `K`, small Krylov space) and each spectrum edge is peeled up front
with an [`EdgeFilter`](@ref) count target.

# Pipeline
1. **Mapping** ([`build_mapping_plan`](@ref)): rescale `H → H̃ = (H-b)/a` into `[-1, 1]`.
2. **DoS** ([`calculate_dos`](@ref)) on the rescaled operator.
3. **Window** ([`resolve_window`](@ref)): the requested window in rescaled coordinates.
4. **Converge + glue** ([`_patched_glued_spectrum`](@ref)): peel each present spectrum edge with an
   `EdgeFilter` + count target, tile the strictly-interior remainder with delta [`Window`](@ref)
   patches, and stitch by ownership (no gaps, no duplicates). Un-rescale to true energies.

# Keywords
- `mc::MappingConfig`, `dosc::DoSConfig`: per-stage configuration.
- `tol`, `eigentol::Real`: Ritz / eigen residual tolerances (true-energy `eigentol`).
- `overestimate_iters::Real`: Krylov-budget safety factor (floored at `2.5` per patch — the
  near-edge brackets of a `Window` converge late).
- `layout::Layout`: Krylov block layout.

Returns `(vals, vecs)` in **true** energies, ascending. (Placeholder return shape; a `PolfedResult`
+ `Report` will replace it once the patched path is finalized.)
"""
function run_patched_polfed_window_pipeline(f!, x0::AbstractVecOrMat, howmany_per_patch::Integer, target;
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

    iv_edge = Interval(-1.0, -0.8)
    filter = EdgeFilter(iv_edge, :both)
    P!      = @stage timer :transform  build_transform(filter, mp)
    eig_target = resolve_eig_target(exposed_end(filter), 50)

    result = @stage timer :lanczos lanczos(P!, mp.sys.x0, eig_target;
                    rot        = Defaults..rot,
                    basistype  = Defaults.polfed.basistype,
                    maxdim     = maxdim,
                    tol        = tol,
                    eigentol   = eigentol,
                    layout     = layout,
                    mapvals    = mp.f!,
                    next_check = Defaults.polfed.conv_checking(expected_dim))

    





    polfed_log(POLFED_INFO_LEVEL, "Starting patched POLFED (window target).",
               target = target, window = (full_iv.left, full_iv.right),
               howmany_per_patch = Int(howmany_per_patch), layout = typeof(layout))

    # 4. Converge + glue the whole window edge-first; result is in RESCALED coordinates, ascending.
    #    `records` is one PatchReport per filtered-Lanczos run (edge peels + interior patches).
    vals, vecs, records = _patched_glued_spectrum(mp, dos, full_iv, howmany_per_patch,
                                                  tol, eigentol, overestimate_iters, layout, timer)

    # Un-rescale the accumulated Ritz values from rescaled [-1, 1] back to true energies.
    @. vals = vals * mp.a + mp.b

    report = PatchedReport(MappingReport(mp, layout), full_iv, Int(howmany_per_patch),
                           Defaults.patched_polfed.howmany_edge, Defaults.patched_polfed.howmany_pad,
                           records, length(vals), mp.a, mp.b, timer)

    polfed_log(POLFED_INFO_LEVEL, "Patched POLFED finished.",
               eigenpairs = length(vals), runs = length(records), walltime_s = total_wall(timer))

    return vals, vecs, report
end
