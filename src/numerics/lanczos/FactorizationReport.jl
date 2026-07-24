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
- `tol`, `residual::Float64`, `converged_tol::Int`: Ritz convergence.
- `eigentol`, `eigenresidual::Float64`, `converged_eigentol::Int`: eigen convergence. Both
  counts are now the same number: the criterion only ever returns pairs that passed
  **both** gates, so "Ritz-converged" and "eigen-converged" no longer name different sets.
- `complete::Bool`: the criterion certified the returned set complete (its stopping rule
  was met), as opposed to the run being handed back at the Krylov budget ceiling. Consumers
  should read this rather than re-deriving completeness from the residuals.
- `numofchecks::Int`, and per-check `converged_history`, `krylovbasisdim_history`, `maxresidual_history`.
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
    complete::Bool
    numofchecks::Int
    converged_history::Vector{Int}
    krylovbasisdim_history::Vector{Int}
    maxresidual_history::Vector{Float64}
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
            reserved_cols = size(factorization.basis.basis, 2)
            needed_cols   = factorization.basis.nvecs
        elseif basistype <: VectorBasis
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
            state.converged,
            criterion.eigentol,
            state.eigenresidual,
            state.converged,
            state.complete,
            state.numofchecks,
            state.converged_history,
            state.krylovbasisdim_history,
            state.maxresidual_history,
            timer,
        )
    end
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

    eig_ok = report.converged_eigentol == report.howmany
    lan_ok = report.converged_tol == report.howmany

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
    println("- Number of converged eigenpairs:   $eig_color (out of $howmany requested)")
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
        header = ["Checking", "Krylov dim.", "Converged", "Residual"]
        data = hcat(1:report.numofchecks, report.krylovbasisdim_history,
                    report.converged_history, report.maxresidual_history)

        println("- Convergence check was performed $(report.numofchecks) times, here is the table of results:")

        pretty_table(
            data;
            formatters    = ft_printf("%d", 1:3),
            header        = header,
            header_crayon = crayon"white bold",   # matches the POLFED report's copy of this table
            tf            = tf_unicode_rounded
        )
    end
end

"""Convenience wrapper: display the report with colors enabled."""
print_factorization_report(report::FactorizationReport; kwargs...) =
    display_factorization_report(report, true; kwargs...)
