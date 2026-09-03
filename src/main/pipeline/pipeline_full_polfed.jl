"""
    run_full_polfed_pipeline(f!, x0, howmany_per_patch; kwargs...)
        -> (vals, vecs, report)

Resolve the full spectrum edge-first. The only persistent eigenvector allocation is the final
`D × D` matrix (or `D × D` row-layout equivalent); edge and interior vectors are written into
it immediately, and the final ordering uses one `D`-element scratch vector.
"""
function run_full_polfed_pipeline(
    f!,
    x0::AbstractVecOrMat,
    howmany_per_patch::Integer;
    mc::MappingConfig = MappingConfig(
        optimize_mapping = Defaults.full_polfed.optimize_mapping,
    ),
    dosc::DoSConfig = DoSConfig(
        N = Defaults.full_polfed.N,
        R = Defaults.full_polfed.R,
        kernel = Defaults.full_polfed.kernel,
    ),
    tol::Real = Defaults.full_polfed.tol,
    eigentol::Real = Defaults.full_polfed.eigentol,
    overestimate_iters::Real = Defaults.full_polfed.overestimate_iters,
    layout::Layout = ColumnLayout(),
    edge_distance::Real = _DEFAULT_EDGE_DISTANCE,
    howmany_pad::Integer = Defaults.full_polfed.howmany_pad,
    max_retries::Integer = Defaults.full_polfed.max_glue_retries,
    retry_growth::Real = Defaults.full_polfed.retry_budget_growth,
)
    howmany_per_patch > 0 || throw(ArgumentError("howmany_per_patch must be positive."))
    timer = StageTimer()
    polfed_log(POLFED_INFO_LEVEL, "full_polfed started (whole rescaled spectrum [-1, 1]).",
               howmany_per_patch = Int(howmany_per_patch), tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    mp.sys.dim > 1 || throw(ArgumentError("full POLFED requires an operator dimension above one."))
    dos = @stage timer :dos calculate_dos(dosc, dos_operator(mp), mp.sys)
    request = Interval(-1.0, 1.0)
    records = PatchReport[]
    mapvals = _make_rescaled(mp.f!, mp.a, mp.b)
    eigentol_rescaled = eigentol / mp.a
    buffer = _dense_result_buffer(
        mp, mp.sys.dim, layout;
        duplicate_atol = _result_duplicate_tolerance(eigentol_rescaled),
    )

    left_edge, right_edge = _resolve_edges(
        mp, dos, request, buffer, records, mapvals, tol, eigentol_rescaled,
        overestimate_iters, layout, timer;
        edge_distance, max_retries, retry_growth,
        algorithm_defaults = Defaults.full_polfed,
    )
    _resolve_interior(
        mp, dos, request, left_edge, right_edge, buffer, records, howmany_per_patch,
        mapvals, tol, eigentol_rescaled, overestimate_iters, layout, timer;
        edge_distance, howmany_pad, max_retries, retry_growth,
        algorithm_defaults = Defaults.full_polfed,
    )

    vals, vecs = _finalize_result!(buffer)
    @. vals = vals * mp.a + mp.b
    length(vals) == mp.sys.dim || polfed_log(POLFED_WARN_LEVEL,
        "Full POLFED did not certify the complete matrix spectrum.",
        expected = mp.sys.dim, returned = length(vals))
    report = PatchedReport(
        MappingReport(mp, layout), request, Int(howmany_per_patch),
        _edge_budget(left_edge, right_edge), Int(howmany_pad), records,
        length(vals), mp.a, mp.b, timer,
    )
    # `vecs` comes back column-oriented from `_finalize_result!` regardless of the Krylov
    # layout the run used, so the result records ColumnLayout.
    return PatchedResult(vals, vecs, report, ColumnLayout())
end
