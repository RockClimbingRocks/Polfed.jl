"""
    run_patched_polfed_target_pipeline(
        f!, x0, howmany, howmany_per_patch, target; kwargs...
    ) -> (vals, vecs, report)

Edge-first patched POLFED for a count target. The final layout-aware eigenvector matrix is
allocated once at `dim × howmany` (or its row-layout transpose); every accepted patch vector
is copied into it immediately. A bounded in-place selection retains the nearest eigenpairs.
"""
function run_patched_polfed_target_pipeline(
    f!,
    x0::AbstractVecOrMat,
    howmany::Integer,
    howmany_per_patch::Integer,
    target;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.polfed.tol,
    eigentol::Real = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters,
    layout::Layout = ColumnLayout(),
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.patched_polfed.howmany_pad,
    max_retries::Integer = Defaults.patched_polfed.max_glue_retries,
    retry_growth::Real = Defaults.patched_polfed.retry_budget_growth,
)
    requested = Int(howmany)
    requested > 0 || throw(ArgumentError("howmany must be positive; got $howmany."))
    howmany_per_patch > 0 || throw(ArgumentError("howmany_per_patch must be positive."))
    spec = normalize_target_spec(target)
    spec isa TargetWindowSpec && throw(ArgumentError(
        "The patched count pipeline requires a point target, not a window.",
    ))
    timer = StageTimer()
    log_run_start("patched_polfed", :count, spec; howmany = requested,
                  howmany_per_patch = Int(howmany_per_patch), tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    dos = @stage timer :dos calculate_dos(dosc, mp.f!_rescaled, mp.sys)
    center = @stage timer :interval resolve_target(spec, requested, dos, mp.a, mp.b)
    request = @stage timer :interval resolve_interval(center, dos)

    records = PatchReport[]
    mapvals = _make_rescaled(mp.f!, mp.a, mp.b)
    eigentol_rescaled = eigentol / mp.a
    buffer = _dense_result_buffer(
        mp, min(requested, mp.sys.dim), layout;
        target = center.rescaled_energy,
        duplicate_atol = _result_duplicate_tolerance(eigentol_rescaled),
    )

    left_edge, right_edge = _resolve_edges(
        mp, dos, request, buffer, records, mapvals, tol, eigentol_rescaled,
        overestimate_iters, layout, timer;
        edge_distance, max_retries, retry_growth,
    )
    _resolve_interior(
        mp, dos, request, left_edge, right_edge, buffer, records, howmany_per_patch,
        mapvals, tol, eigentol_rescaled, overestimate_iters, layout, timer;
        edge_distance, howmany_pad, max_retries, retry_growth,
    )

    vals, vecs = _finalize_result!(buffer)
    @. vals = vals * mp.a + mp.b
    length(vals) < requested && polfed_log(POLFED_WARN_LEVEL,
        "Patched POLFED returned fewer eigenpairs than requested.",
        requested = requested, converged = length(vals))
    report = PatchedReport(
        MappingReport(mp, layout), request, Int(howmany_per_patch),
        _edge_budget(left_edge, right_edge), Int(howmany_pad), records,
        length(vals), mp.a, mp.b, timer,
    )
    # `vecs` comes back column-oriented from `_finalize_result!` regardless of the Krylov
    # layout the run used, so the result records ColumnLayout.
    return PatchedResult(vals, vecs, report, ColumnLayout())
end
