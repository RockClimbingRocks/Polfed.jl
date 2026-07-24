# The `polfed` pipeline. Given an operator `f!`, a seed/block `x0`, a count and a target, it walks
# the stages — mapping → DoS → target → interval → filter → transform → Lanczos → post-process — and
# returns a `PolfedResult`. The public `polfed` entrypoints in `entrypoint/` are thin wrappers that
# prepare inputs (e.g. the matrix path builds fast row-layout kernels) and call this. Each entrypoint
# function has its OWN pipeline (`run_<name>_pipeline`); e.g. `patched_polfed` → `run_patched_polfed_pipeline`.

"""
    run_polfed_target_pipeline(f!, x0, howmany, target; mc, dosc, tol, eigentol, overestimate_iters, layout) -> PolfedResult

The POLFED pipeline: compute `howmany` eigenpairs of the operator `f!` near a target spectral
region by polynomial-filtered Lanczos. The problem shape (dimension, block size, element type,
device) is inferred from `x0` and carried on the mapping plan — no `dim`/`eltype`/`blocksize`
arguments. Called by [`polfed`](@ref); not a public entrypoint itself. Each entrypoint function has
its own `run_<name>_pipeline`.

# Pipeline
1. **Mapping** ([`build_mapping_plan`](@ref)): rescale `H → H̃ = (H-b)/a` into `[-1, 1]`
   (estimating `Emin`/`Emax` with a short Lanczos probe if not given).
2. **DoS** ([`calculate_dos`](@ref)): estimate the density of states of the *rescaled* operator.
   Layout-agnostic — the moments are spectral, so the row/column orientation of `mp.f!_rescaled`
   is irrelevant for a Hermitian operator.
3. **Target** ([`resolve_target`](@ref)): resolve the requested region into a rescaled target center.
4. **Interval** ([`resolve_interval`](@ref)): grow the center into the rescaled window that holds
   `howmany` states, using the DoS.
5. **Filter + Transform** ([`resolve_filter`](@ref) → [`build_transform`](@ref)): pick/build/validate
   the filter from where the window sits, then build `f!_transformed`, the Clenshaw filter of `H̃`,
   which maps the target window to a spectral extreme.
6. **Lanczos** ([`lanczos`](@ref)) on `f!_transformed`, with `mapvals = mp.f!` (the *original*
   operator) so the returned Ritz values are **true** energies of `H` (Rayleigh quotients). The
   extreme to select follows [`exposed_end`](@ref)`(filter)`; the Krylov budget follows
   [`expected_krylov_dim`](@ref)`(filter, …)`.
7. **Post-process** ([`build_polfed_result`](@ref)): filter to the target interval, sort wrt `H`,
   warn on shortfall, and assemble the [`Report`](@ref).

# Keywords
- `mc::MappingConfig`, `dosc::DoSConfig`: per-stage configuration.
- `tol::Real`, `eigentol::Real`: Ritz-residual / eigen-residual tolerances.
- `overestimate_iters::Real`: Krylov-budget safety factor.
- `layout::Layout = ColumnLayout()`: the Krylov block layout Lanczos runs in. Column for a plain
  function `f!` (operator action `H·X`); the matrix entrypoint calls with `RowLayout()` after
  converting `x0` and the operators to row form (`X·H`). Pipeline-internal — not a public knob.

The reorthogonalization technique and Krylov basis storage are fixed (`Defaults.polfed.rot`,
`Defaults.polfed.basistype`); the eigenvalue selection and Krylov dimension are derived from the filter.
"""
function run_polfed_target_pipeline(f!, x0::AbstractVecOrMat, howmany::Integer, target;
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

    polfed_log(POLFED_INFO_LEVEL, "Starting POLFED (count target).",
               howmany = howmany, target = target, layout = typeof(layout),
               tol = tol, eigentol = eigentol, overestimate_iters = overestimate_iters)

    # 1. Mapping: rescale H → H̃ = (H - b)/a into [-1, 1] (estimating Emin/Emax if needed).
    mp = @stage timer :bounds build_mapping_plan(mc, f!, x0)

    # 2. DoS on the rescaled operator (layout-agnostic — moments are spectral).
    dos = @stage timer :dos calculate_dos(dosc, mp.f!_rescaled, mp.sys)

    # 3–4. Resolve the rescaled target center, then grow it into the exposed window. Both record
    #      under :interval — one stage meaning "arrive at the rescaled target interval" — so this
    #      pipeline's breakdown lines up with the window pipeline, where `resolve_window` does the
    #      same job in a single call. Same-name stages accumulate, so :interval reads as their sum.
    rescaled_target = @stage timer :interval resolve_target(spec, howmany, dos, mp.a, mp.b)
    target_interval = @stage timer :interval resolve_interval(rescaled_target, dos)

    # 5. Filter + Transform: pick/build/validate the filter, then build the Clenshaw filter of H̃.
    filter  = @stage timer :filter     resolve_filter(target_interval)
    P!      = @stage timer :transform  build_transform(filter, mp)

    # 6. Lanczos on the transformed operator; target eigenvalues are extremal so convergence is
    #    fast. `mapvals = mp.f!` maps the Ritz values back to TRUE energies of H.
    #    Convergence checks are deferred until ~the expected (un-inflated) Krylov dim via
    #    `next_check` (before then no target eigenpair can have converged — see `conv_checking`).
    eig_target   = resolve_eig_target(exposed_end(filter), howmany)
    expected_dim = expected_krylov_dim(filter, Int(howmany), mp.sys.block_size, 1.0)               # un-inflated
    maxdim       = expected_krylov_dim(filter, Int(howmany), mp.sys.block_size, overestimate_iters) # inflated budget

    polfed_log(POLFED_INFO_LEVEL, "Running Lanczos on the transformed operator.",
               eig_target = typeof(eig_target), expected_krylov_dim = expected_dim,
               maxdim = maxdim, block_size = mp.sys.block_size)
    result = @stage timer :lanczos lanczos(P!, mp.sys.x0, eig_target;
                     rot        = Defaults.polfed.rot,
                     basistype  = Defaults.polfed.basistype,
                     maxdim     = maxdim,
                     tol        = tol,
                     eigentol   = eigentol,
                     layout     = layout,
                     mapvals    = mp.f!,
                     next_check = Defaults.polfed.conv_checking(expected_dim))

    polfed_log(POLFED_INFO_LEVEL, "Lanczos finished.",
               converged = result.converged, iterations = result.report.itersneeded,
               walltime_s = total_wall(timer))

    # 7. Post-process the raw Lanczos result into the returned PolfedResult: filter to the target
    #    interval, sort wrt H (not the transformed P(H)), warn on shortfall, and assemble the Report.
    return build_polfed_result(result, mp, target_interval, spec, filter, layout, Int(howmany), timer)
end
