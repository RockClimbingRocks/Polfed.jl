
double_polfed(f!, x0::AbstractVecOrMat, howmany::Integer, target;
       mc::MappingConfig        = MappingConfig(),
       dosc::DoSConfig          = DoSConfig(),
       tol::Real                = Defaults.polfed.tol,
       eigentol::Real           = Defaults.polfed.eigentol,
       overestimate_iters::Real = Defaults.polfed.overestimate_iters) =
    run_polfed_target_pipeline(f!, x0, howmany, target;
                 mc = mc, dosc = dosc, tol = tol, eigentol = eigentol,
                 overestimate_iters = overestimate_iters)   # column layout (default)





function double_polfed(H::AbstractMatrix, x0::AbstractVecOrMat, howmany::Integer, target;
    mc::MappingConfig        = MappingConfig(),
    dosc::DoSConfig          = DoSConfig(),
    tol::Real                = Defaults.polfed.tol,
    eigentol::Real           = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters
)

    f!, x0_row, mc_row = _prepare_matrix_row_pipeline(H, x0, mc)
    return run_polfed_target_pipeline(f!, x0_row, howmany, target;
                        mc = mc_row, dosc = dosc, tol = tol, eigentol = eigentol,
                        overestimate_iters = overestimate_iters, layout = RowLayout())
end





polfed(f!, x0::AbstractVecOrMat, target;
       mc::MappingConfig        = MappingConfig(),
       dosc::DoSConfig          = DoSConfig(),
       tol::Real                = Defaults.polfed.tol,
       eigentol::Real           = Defaults.polfed.eigentol,
       overestimate_iters::Real = Defaults.polfed.overestimate_iters) =
    run_polfed_window_pipeline(f!, x0, target;
                 mc = mc, dosc = dosc, tol = tol, eigentol = eigentol,
                 overestimate_iters = overestimate_iters)   # column layout (default)


function polfed(H::AbstractMatrix, x0::AbstractVecOrMat, target;
    mc::MappingConfig        = MappingConfig(),
    dosc::DoSConfig          = DoSConfig(),
    tol::Real                = Defaults.polfed.tol,
    eigentol::Real           = Defaults.polfed.eigentol,
    overestimate_iters::Real = Defaults.polfed.overestimate_iters
)

    f!, x0_row, mc_row = _prepare_matrix_row_pipeline(H, x0, mc)
    return run_polfed_window_pipeline(f!, x0_row, target;
                        mc = mc_row, dosc = dosc, tol = tol, eigentol = eigentol,
                        overestimate_iters = overestimate_iters, layout = RowLayout())
end
