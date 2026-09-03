"""
    run_patched_polfed_window_pipeline(
        f!, x0, howmany_per_patch, target; kwargs...
    ) -> (vals, vecs, report)

Edge-first patched POLFED for an explicit window. Patch eigenvectors are copied out before
the next Lanczos run, so no result view keeps an old patch matrix alive.
"""
function run_patched_polfed_window_pipeline(
    f!,
    x0::AbstractVecOrMat,
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
    howmany_per_patch > 0 || throw(ArgumentError("howmany_per_patch must be positive."))
    spec = normalize_target_spec(target)
    spec isa TargetWindowSpec || throw(ArgumentError(
        "The patched window pipeline requires an explicit window target.",
    ))
    timer = StageTimer()
    log_run_start("patched_polfed", :window, spec;
                  howmany_per_patch = Int(howmany_per_patch), tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    dos = @stage timer :dos calculate_dos(dosc, dos_operator(mp), mp.sys)
    request = @stage timer :interval resolve_window(spec, mp.a, mp.b)

    records = PatchReport[]
    mapvals = _make_rescaled(mp.f!, mp.a, mp.b)
    eigentol_rescaled = eigentol / mp.a
    buffer = _window_result_buffer(
        mp, layout; duplicate_atol = _result_duplicate_tolerance(eigentol_rescaled),
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
    report = PatchedReport(
        MappingReport(mp, layout), request, Int(howmany_per_patch),
        _edge_budget(left_edge, right_edge), Int(howmany_pad), records,
        length(vals), mp.a, mp.b, timer,
    )
    # `vecs` comes back column-oriented from `_finalize_result!` regardless of the Krylov
    # layout the run used, so the result records ColumnLayout.
    return PatchedResult(vals, vecs, report, ColumnLayout())
end
