"""
    run_polfed_window_pipeline(f!, x0, target; kwargs...) -> PolfedResult

Single-filter POLFED for an explicit spectral window.
"""
function run_polfed_window_pipeline(
    f!,
    x0::AbstractVecOrMat,
    target;
    mc::MappingConfig = MappingConfig(),
    dosc::DoSConfig = DoSConfig(),
    tol::Real = Defaults.polfed.tol,
    eigentol::Real = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters,
    layout::Layout = ColumnLayout(),
)
    timer = StageTimer()
    spec = normalize_target_spec(target)
    spec isa TargetWindowSpec || throw(ArgumentError(
        "The window pipeline requires (:rescaled, (left, right)), " *
        "(:unrescaled, (left, right)), or a bare (left, right).",
    ))
    log_run_start("polfed", :window, spec; tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    dos = @stage timer :dos calculate_dos(dosc, dos_operator(mp), mp.sys)
    interval = @stage timer :interval resolve_window(spec, mp.a, mp.b)
    howmany = howmany_in_interval(dos, interval)
    filter = @stage timer :filter resolve_filter(interval)
    transformed = @stage timer :transform build_transform(filter, mp)
    mapvals = _make_rescaled(mp.f!, mp.a, mp.b)
    eigentol_rescaled = eigentol / mp.a
    expected_dim = expected_krylov_dim(filter, howmany, mp.sys.block_size, 1.0)
    maxdim = expected_krylov_dim(
        filter, howmany, mp.sys.block_size, max(2.5, overestimate_iters),
    )

    eig_target = window_eig_target(filter, interval)
    log_prelanczos(mp, filter, eig_target, howmany, expected_dim, maxdim, tol, eigentol_rescaled;
                   interval = interval, rescaled_values = true)
    result = @stage timer :lanczos _filtered_lanczos(
        transformed, mp.sys.x0, eig_target;
        rot = Defaults.polfed.rot,
        basistype = Defaults.polfed.basistype,
        maxdim = maxdim,
        tol = tol,
        eigentol = eigentol_rescaled,
        layout = layout,
        mapvals = mapvals,
        next_check = Defaults.polfed.conv_checking(expected_dim),
    )
    log_lanczos_outcome(result)
    return build_polfed_result(
        result, mp, interval, spec, filter, layout, howmany, timer;
        vals_rescaled = true,
    )
end
