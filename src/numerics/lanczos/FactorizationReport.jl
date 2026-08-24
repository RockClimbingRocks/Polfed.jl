"""
    FactorizationReport(criterion::ConvergenceCriterion, state::ConvergenceState,
                        factorization::KrylovFactorization, iterator::LanczosIterator,
                        timer::StageTimer)

Detailed record of a single factorization run: factorization/basis statistics,
convergence results, and stage timings. Built internally after a Lanczos or
block-Lanczos run; consumed by [`display_factorization_report`](@ref).

# Fields
- `factorizationtype::String`, `blocksize::Int`, `basistype::Type{<:OrthonormalBasis}`.
- `ncols_{cpu,gpu}_{reserved,needed}::Int`: basis columns reserved/used per device.
- `howmany::Int`: number of targeted eigenpairs (`ntargets(target)`).
- `itersneeded::Int`, `itersreserved::Int`: iterations used vs. budgeted.
- `rot::ReOrthTechnique`.
- `tol`, `residual::Float64`, `converged_tol::Int`: Ritz convergence — worst Ritz residual and
  how many pairs pass that gate.
- `eigentol`, `eigenresidual::Float64`, `converged_eigentol::Int`: eigen convergence — worst
  eigen residual over the RETURNED pairs, and how many of the pairs examined pass that gate.
- `converged::Int`: pairs actually returned. This is a contiguous run, so it is bounded by both
  counts above and usually smaller than either. The two gaps say different things:
  `converged_tol > converged` means `eigentol`, not the Krylov budget, cut the run — the Ritz
  residual is at its floor precisely because the Krylov space converged, and no extra iteration
  changes that. `converged_eigentol > converged` means the contiguity invariant then threw away
  pairs that had passed both gates, because a failure sits between them and the extreme.
- `eigenworst::Float64`: worst eigen residual over everything EXAMINED (failures included), as
  opposed to `eigenresidual` over what was kept. This is the number that says how far
  `eigentol` would have to move to admit the whole examined set.
- `complete::Bool`: the criterion certified the returned set complete (its stopping rule
  was met), as opposed to the run being handed back at the Krylov budget ceiling. Consumers
  should read this rather than re-deriving completeness from the residuals.
- `numofchecks::Int`, and per-check `converged_history`, `ritzconverged_history`,
  `eigenconverged_history` (`-1` where nothing was reconstructed), `krylovbasisdim_history`,
  `maxresidual_history`, `eigenresidual_history` (the worst eigen residual *examined* at that
  check, `NaN` where the cheap pre-gate skipped reconstruction), `checktime_history` (wall
  seconds that check took).
- `timer::StageTimer`: per-stage timings accumulated over all iterations, addressed by NAME —
  `:tridiag`, `:matvec`, `:overlap`, `:orthog`, `:reorthog`, `:conv_check`. Read as
  `timer[:matvec]` (wall s), `timer[:matvec, :cpu]`, `timer[:matvec, :n]` (iterations that hit it).
"""
mutable struct FactorizationReport
    factorizationtype::String
    blocksize::Int
    basistype::Type{<:OrthonormalBasis}
    ncols_cpu_reserved::Int
    ncols_gpu_reserved::Int
    ncols_cpu_needed::Int
    ncols_gpu_needed::Int
    howmany::Int
    itersneeded::Int
    itersreserved::Int
    rot::ReOrthTechnique
    tol::Float64
    residual::Float64
    converged_tol::Int
    eigentol::Float64
    eigenresidual::Float64
    converged_eigentol::Int
    converged::Int
    eigenworst::Float64
    complete::Bool
    numofchecks::Int
    converged_history::Vector{Int}
    ritzconverged_history::Vector{Int}
    eigenconverged_history::Vector{Int}
    krylovbasisdim_history::Vector{Int}
    maxresidual_history::Vector{Float64}
    eigenresidual_history::Vector{Float64}
    checktime_history::Vector{Float64}
    timer::StageTimer

    """Build `FactorizationReport` from convergence criterion/state and factorization runtime state."""
    function FactorizationReport(
        criterion::ConvergenceCriterion,
        state::ConvergenceState,
        factorization::KrylovFactorization,
        iterator::LanczosIterator,
        timer::StageTimer,
    )
        factorizationtype_ = string(typeof(factorization))
        factorizationtype  = split(first(split(factorizationtype_, "{")), ".")[end]
        blocksize = isa(factorization, LanczosFactorization) ? 1 : factorization.blocksize
        basistype = typeof(factorization.basis)

        # Reserved/used basis columns, normalized so both backends report column counts
        # (MatrixBasis.nvecs already counts columns; VectorBasis.nvecs counts blocks).
        if basistype <: MatrixBasis
            basis = factorization.basis
            reserved_cols = basis.layout isa RowLayout ? size(basis.basis, 1) :
                                                         size(basis.basis, 2)
            needed_cols   = factorization.basis.nvecs
        elseif basistype <: VectorBasis
            # A VectorBasis grows on demand, so nothing is reserved beyond what is used:
            # `reserved == needed` here by construction, unlike the preallocated MatrixBasis
            # where the gap between them is the budget that went unspent.
            bs = factorization.basis.blocksize
            reserved_cols = length(factorization.basis.basis) * bs
            needed_cols   = factorization.basis.nvecs * bs
        else
            error("Unsupported basis type: $basistype")
        end

        if factorization.pu isa GPU
            ncols_gpu_reserved, ncols_cpu_reserved = reserved_cols, 0
            ncols_gpu_needed, ncols_cpu_needed     = needed_cols, 0
        else
            ncols_cpu_reserved, ncols_gpu_reserved = reserved_cols, 0
            ncols_cpu_needed, ncols_gpu_needed     = needed_cols, 0
        end

        rc = requested_count(criterion.target)
        howmany = rc >= 0 ? rc : state.converged   # value-based targets: dynamic count

        new(
            factorizationtype,
            blocksize,
            basistype,
            ncols_cpu_reserved,
            ncols_gpu_reserved,
            ncols_cpu_needed,
            ncols_gpu_needed,
            howmany,
            state.numiter,
            criterion.maxiter,
            iterator.rot,
            criterion.tol,
            state.residual,
            state.ritzconverged,
            criterion.eigentol,
            state.eigenresidual,
            state.eigenconverged,
            state.converged,
            state.eigenworst,
            state.complete,
            state.numofchecks,
            state.converged_history,
            state.ritzconverged_history,
            state.eigenconverged_history,
            state.krylovbasisdim_history,
            state.maxresidual_history,
            state.eigenresidual_history,
            state.checktime_history,
            timer,
        )
    end
end

"""
    CONVERGENCE_TABLE_HEADER

Column titles of the per-check convergence-history table, shared by the `FactorizationReport`
display and the POLFED `Report`'s copy of the same table.

The two residuals come first (the signals), then the two gate counts as `ritz/eigen`, then the
contiguous run those counts collapse to. Read across, a stalled run explains itself: `Ritz res.`
at the machine floor with `Eig. res.` above `eigentol`, `#Ritz/Eig ok` frozen, and
`Converged` below both, is a run no extra iteration will improve. The gap between the eigen
count and `Converged` is the contiguity invariant discarding pairs that individually passed but
sit behind a failure. An eigen count of `-1` renders as `-`: nothing was reconstructed.

`Time [s]` closes the row with what that check cost. Reconstruction dominates a POLFED run once
the target count is large, so a column of identical counts beside a column of seconds is the
whole argument for checking less often.
"""
const CONVERGENCE_TABLE_HEADER =
    ["Check", "Kry. Dim.", "Ritz res.", "Eig. res.", "#Ritz/Eig ok", "Converged", "Time [s]"]

"""
    convergence_table_rows(report::FactorizationReport) -> Matrix{String}

The per-check convergence history as a `numofchecks × 7` matrix of formatted strings, matching
[`CONVERGENCE_TABLE_HEADER`](@ref): check index, Krylov dimension, worst Ritz residual, worst
eigen residual examined, the two gate counts as `ritz/eigen`, the contiguous run returned, and
the wall seconds that check took.

Unmeasured entries render as `-` (residual `NaN`, count `-1`): that check's cheap pre-gate fired
before any eigenvector was reconstructed, so nothing was measured — different from measuring
zero, and worth telling apart when reading where the cost went.

# Notes
Pre-formatted here rather than at the sink so that both table backends (the built-in plain
renderer and the optional PrettyTables one) receive identical cells.
"""
function convergence_table_rows(report::FactorizationReport)
    n = report.numofchecks
    rows = Matrix{String}(undef, n, 7)
    for i in 1:n
        eig  = get(report.eigenresidual_history, i, NaN)
        neig = get(report.eigenconverged_history, i, -1)
        secs = get(report.checktime_history, i, NaN)
        rows[i, 1] = string(i)
        rows[i, 2] = string(report.krylovbasisdim_history[i])
        rows[i, 3] = @sprintf("%.2e", report.maxresidual_history[i])
        rows[i, 4] = isnan(eig) ? "-" : @sprintf("%.2e", eig)
        rows[i, 5] = string(get(report.ritzconverged_history, i, 0), "/", neig < 0 ? "-" : string(neig))
        rows[i, 6] = string(report.converged_history[i])
        rows[i, 7] = isnan(secs) ? "-" : @sprintf("%.3f", secs)
    end
    return rows
end

"""
    display_factorization_report(report::FactorizationReport, use_colors::Bool;
                                 show_convergence_details::Bool=false, show_timings::Bool=true)

Print a human-readable summary of a `FactorizationReport`: convergence counts,
iteration usage, basis/reorthogonalization info, timings, and (optionally) the
per-check convergence table.
"""
function display_factorization_report(report::FactorizationReport, use_colors::Bool;
                                      show_convergence_details::Bool=false,
                                      show_timings::Bool=true)

    f = Formatter(use_colors)

    conv_ok = report.converged == report.howmany
    eig_ok  = report.converged_eigentol >= report.howmany   # both gate counts may reach past
    lan_ok  = report.converged_tol >= report.howmany        # the request; green at "enough"

    conv_color = conv_ok ? green(f, string(report.converged)) : red(f, string(report.converged))
    eig_color = eig_ok ? green(f, string(report.converged_eigentol)) : red(f, string(report.converged_eigentol))
    lan_color = lan_ok ? green(f, string(report.converged_tol)) : red(f, string(report.converged_tol))

    remaining_percentage = (1 - report.itersneeded / report.itersreserved) * 100
    remaining_color = remaining_percentage <= 3   ? red(f, @sprintf("%.2f%%", remaining_percentage)) :
                      remaining_percentage <= 5   ? yellow(f, @sprintf("%.2f%%", remaining_percentage)) :
                      remaining_percentage <= 25  ? green(f, @sprintf("%.2f%%", remaining_percentage)) :
                                                    red(f, @sprintf("%.2f%%", remaining_percentage))

    t = report.timer
    total_walltime = total_wall(t)
    walltime, units_wt = total_walltime <= 60*2     ? (total_walltime, "seconds") :
                         total_walltime <= 60*60*2  ? (total_walltime/60, "minutes") :
                                                      (total_walltime/3600, "hours")

    total_cputime = total_cpu(t)
    cputime, units_ct = total_cputime <= 60*2     ? (total_cputime, "seconds") :
                        total_cputime <= 60*60*2  ? (total_cputime/60, "minutes") :
                                                    (total_cputime/3600, "hours")

    # (Mapping, Reorthogonalization, Convergence check, others) — read by stage NAME.
    _pct(x, tot) = tot > 0 ? x / tot * 100 : 0.0
    perc_wt_grouped = [_pct(t[:matvec], total_walltime), _pct(t[:reorthog], total_walltime),
                       _pct(t[:conv_check], total_walltime),
                       _pct(t[:tridiag] + t[:overlap] + t[:orthog], total_walltime)]
    perc_ct_grouped = [_pct(t[:matvec, :cpu], total_cputime), _pct(t[:reorthog, :cpu], total_cputime),
                       _pct(t[:conv_check, :cpu], total_cputime),
                       _pct(t[:tridiag, :cpu] + t[:overlap, :cpu] + t[:orthog, :cpu], total_cputime)]

    formatted_perc_wt = join([cyan(f, @sprintf("%.2f%%", p)) for p in perc_wt_grouped], ", ")
    formatted_perc_ct = join([cyan(f, @sprintf("%.2f%%", p)) for p in perc_ct_grouped], ", ")

    fac_type = bold(f, string(report.factorizationtype))
    blocksize = bold(f, string(report.blocksize))
    howmany   = cyan(f, string(report.howmany))
    tol       = cyan(f, @sprintf("%.2e", report.tol))
    eigentol  = cyan(f, @sprintf("%.2e", report.eigentol))
    residual  = yellow(f, @sprintf("%.2e", report.residual))
    eigrnres  = yellow(f, @sprintf("%.2e", report.eigenresidual))
    itersneeded   = bold(f, string(report.itersneeded))
    itersreserved = cyan(f, string(report.itersreserved))
    total_wt      = cyan(f, @sprintf("%.2f %s", walltime, units_wt))
    total_ct      = cyan(f, @sprintf("%.2f %s", cputime, units_ct))

    completeness = report.complete ? green(f, "certified complete") :
                                     yellow(f, "NOT certified — budget exhausted, set may be short")

    println(blue(f, "Factorization Report:"), " ($fac_type with blocksize $blocksize)")
    println("- Number of converged eigenpairs:   $conv_color (out of $howmany requested)")
    println("- Returned set:                     $completeness")
    println("- Lanczos convergence satisfied by: $lan_color (with tolerance $tol, max residual $residual)")
    println("- Eigen convergence satisfied by:   $eig_color (with tolerance $eigentol, max residual $eigrnres)")
    println("- Iterations needed: $itersneeded (out of $itersreserved reserved, overestimated by $remaining_color)")

    if show_timings
        println(blue(f, "Timings:"), " Percentages are distributed as: (Mapping, Reorthogonalization, Convergence check, others)")
        println("- Walltime of factorization took: $total_wt ($formatted_perc_wt)")
        println("- CPU time of factorization took: $total_ct ($formatted_perc_ct)")
    end

    if show_convergence_details
        println("- Convergence check was performed $(report.numofchecks) times, here is the table of results:")
        render_detail_table(stdout, CONVERGENCE_TABLE_HEADER, convergence_table_rows(report);
                            bold_header = use_colors)
    end
end

"""Convenience wrapper: display the report with colors enabled."""
print_factorization_report(report::FactorizationReport; kwargs...) =
    display_factorization_report(report, true; kwargs...)
