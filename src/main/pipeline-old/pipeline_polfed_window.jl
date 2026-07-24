
"""
    run_polfed_window_pipeline(f!, x0, target; mc, dosc, tol, eigentol, overestimate_iters, layout) -> PolfedResult

      The window-target POLFED pipeline: compute **all** eigenpairs of `f!` whose true energy lies in the
      window `target`. `target` must normalize to a [`TargetWindowSpec`](@ref) (`(:rescaled,(lo,hi))`,
      `(:unrescaled,(lo,hi))`, or a bare `(lo,hi)`); a point spec throws. Called by [`polfed`](@ref).

      # Pipeline
      1. **Mapping** ([`build_mapping_plan`](@ref)) — rescale `H → H̃ = (H-b)/a` into `[-1,1]`.
      2. **DoS** ([`calculate_dos`](@ref)) — density of the *rescaled* operator.
      3. **Window** ([`resolve_window`](@ref)) — the window spec straight to the rescaled [`Interval`](@ref)
         (rescaling + validation only; no growth).
      4. **Count** ([`howmany_in_interval`](@ref)) — estimate the in-window state count from the DoS (+ a
         guard buffer) to *budget* the Krylov space. 
      5. **Filter + Transform** ([`resolve_filter`](@ref) → [`build_transform`](@ref)) — as the target pipeline.
      6. **Lanczos** ([`lanczos`](@ref)) with a [`ToValue`](@ref) eig-target at the filter cutoff
         ([`window_eig_target`](@ref)): in transformed coordinates the window is extremal, so "all Ritz
         past the cutoff converged (+ completeness guard)" ⇔ "all in-window converged". `mapvals = mp.f!`
         maps the selected Ritz values back to TRUE energies of `H`.
      7. **Post-process** ([`build_polfed_result`](@ref)) — filter to the interval, sort wrt `H`, Report.

      Keywords match [`run_polfed_target_pipeline`](@ref).
"""
function run_polfed_window_pipeline(f!, x0::AbstractVecOrMat, target;
                      mc::MappingConfig        = MappingConfig(),
                      dosc::DoSConfig          = DoSConfig(),
                      tol::Real                = Defaults.polfed.tol,
                      eigentol::Real           = Defaults.polfed.eigentol,
                      overestimate_iters::Real = Defaults.polfed.overestimate_iters,
                      layout::Layout           = ColumnLayout())

    # POLFED-level stage timings. The EXPLICIT 3-argument `@stage timer :name expr` form is used
    # (not the ambient `@stages` block): the whole pipeline is one function body, so there is no
    # parameter to thread away, and `@stages` would wrap it in a `try` — whose scope would hide every
    # variable the pipeline defines from the `build_polfed_result` call at the end.
    timer = StageTimer()

    spec = normalize_target_spec(target)
    spec isa TargetWindowSpec ||
        throw(ArgumentError("run_polfed_window_pipeline requires a window target " *
                            "((:rescaled,(lo,hi)), (:unrescaled,(lo,hi)), or a bare (lo,hi)); " *
                            "got $(typeof(spec)). Use the count-based polfed(f!, x0, howmany, target)."))

    polfed_log(POLFED_INFO_LEVEL, "Starting POLFED (window target).",
               target = target, layout = typeof(layout),
               tol = tol, eigentol = eigentol, overestimate_iters = overestimate_iters)

    # 1. Mapping: rescale H → H̃ = (H - b)/a into [-1, 1] (estimating Emin/Emax if needed).
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0)

    # 2. DoS on the rescaled operator (layout-agnostic — moments are spectral).
    dos = @stage timer :dos calculate_dos(dosc, mp.f!_rescaled, mp.sys)

    # 3. Window → rescaled Interval directly (no DoS growth; rescale + validate only).
    target_interval = @stage timer :interval resolve_window(spec, mp.a, mp.b)

    # 4. Budget the Krylov space: estimate the in-window state count from the DoS (+ guard buffer).
    #    This only SIZES the space; the stopping rule is the ToValue criterion in step 6.
    howmany = howmany_in_interval(dos, target_interval)

    polfed_log(POLFED_INFO_LEVEL, "Window resolved; Krylov space budgeted from the DoS state count.",
               left = target_interval.left, right = target_interval.right,
               estimated_states = howmany)

    # 5. Filter + Transform: pick/build/validate the filter, then build the Clenshaw filter of H̃.
    filter  = @stage timer :filter     resolve_filter(target_interval)
    P!      = @stage timer :transform  build_transform(filter, mp)

    # 6. Lanczos with a Window (mapped-value straddle) criterion — converges a contiguous run
    #    covering the interval plus a bracket past each edge. Those near-boundary brackets converge
    #    LATE (at the uniform tolerance), lagging the in-window states by ~2× the count-based Krylov
    #    estimate, so maxdim is budgeted generously (`≥ 2.5×`). The criterion stops early once the
    #    run straddles both edges, so extra headroom costs basis memory, not iterations.
    #    A `Window` is tested against whatever `mapvals` produces, and `window_eig_target` states
    #    the window in RESCALED coordinates (the space the interval and the filter live in), so
    #    `mapvals` must be rescaled too. It is built from the GENERIC `mp.f!` rather than reusing
    #    `mp.f!_rescaled`: in the matrix entrypoint the latter is the fast value-grouped block
    #    kernel, which cannot be applied to the strided `1×D` eigenvector views the map-back uses.
    #    `build_polfed_result` un-rescales once, at the result boundary.
    eig_target   = window_eig_target(filter, target_interval)
    mapvals      = _make_rescaled(mp.f!, mp.a, mp.b)
    # `eigentol` is a TRUE-energy tolerance, but the residual `mapvals` produces is rescaled:
    # ‖H̃·v − λ̃·v‖ = ‖H·v − λ·v‖ / a. Divide by `a` so the acceptance test still means
    # "‖H·v − λ·v‖ < eigentol"; `build_polfed_result` scales the residuals back for the user.
    eigentol_r   = eigentol / mp.a
    expected_dim = expected_krylov_dim(filter, howmany, mp.sys.block_size, 1.0)
    maxdim       = expected_krylov_dim(filter, howmany, mp.sys.block_size, max(2.5, overestimate_iters))
    result = @stage timer :lanczos lanczos(P!, mp.sys.x0, eig_target;
                     rot        = Defaults.polfed.rot,
                     basistype  = Defaults.polfed.basistype,
                     maxdim     = maxdim,
                     tol        = tol,
                     eigentol   = eigentol_r,
                     layout     = layout,
                     mapvals    = mapvals,
                     next_check = Defaults.polfed.conv_checking(expected_dim))

    polfed_log(POLFED_INFO_LEVEL, "Lanczos finished.",
               converged = result.converged, iterations = result.report.itersneeded,
               walltime_s = total_wall(timer))

    # 7. Post-process into the returned PolfedResult (un-rescale to true energy, filter to
    #    interval, sort wrt H, Report).
    return build_polfed_result(result, mp, target_interval, spec, filter, layout, howmany, timer;
                               vals_rescaled = true)
end
