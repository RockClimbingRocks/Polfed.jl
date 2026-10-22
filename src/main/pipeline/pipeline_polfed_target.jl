"""
    run_polfed_target_pipeline(f!, x0, howmany, target; kwargs...) -> PolfedResult

Single-filter POLFED for a count target.
"""
function run_polfed_target_pipeline(
    f!,
    x0::AbstractVecOrMat,
    howmany::Integer,
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
    spec isa TargetWindowSpec && throw(ArgumentError(
        "The count pipeline requires a point target, not a window.",
    ))
    log_run_start("polfed", :count, spec; howmany = Int(howmany), tol = tol, eigentol = eigentol)
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0; layout = layout)
    dos = @stage timer :dos calculate_dos(dosc, dos_operator(mp), mp.sys)
    center = @stage timer :interval resolve_target(spec, howmany, dos, mp.a, mp.b)
    interval = @stage timer :interval resolve_interval(center, dos)
    filter = @stage timer :filter resolve_filter(interval)
    transformed = @stage timer :transform build_transform(filter, mp)

    expected_dim = expected_krylov_dim(filter, Int(howmany), mp.sys.block_size, 1.0)
    maxdim = expected_krylov_dim(
        filter, Int(howmany), mp.sys.block_size, overestimate_iters,
    )
    eig_target = resolve_eig_target(exposed_end(filter), howmany)
    log_prelanczos(mp, filter, eig_target, howmany, expected_dim, maxdim, tol, eigentol;
                   interval = interval)
    result = @stage timer :lanczos _filtered_lanczos(
        transformed, mp.sys.x0, eig_target;
        rot = Defaults.polfed.rot,
        basistype = Defaults.polfed.basistype,
        maxdim = maxdim,
        tol = tol,
        eigentol = eigentol,
        layout = layout,
        mapvals = mp.f!,
        next_check = Defaults.polfed.conv_checking(expected_dim),
    )
    log_lanczos_outcome(result)
    # `center` (the resolved Target) carries the rescaled target; the result selection needs it in
    # true energy, since that is the point the returned `howmany` pairs are nearest to.
    return build_polfed_result(
        result, mp, interval, spec, filter, layout, Int(howmany), timer;
        target_energy = mp.a * center.rescaled_energy + mp.b,
    )
end
