mutable struct FactorizationReport
    # Factorization info
    factorizationtype::String
    blocksize::Integer
    # Basis related stuff
    basistype::Type{<:OrthonormalBasis}
    ncols_cpu_reserved::Integer
    ncols_gpu_reserved::Integer
    ncols_cpu_needed::Integer
    ncols_gpu_needed::Integer
    howmany::Integer
    itersneeded::Integer
    itersreserved::Integer
    rot::ReOrthTechnique
    # Lanczos convergence
    tol::Real
    residual::Real
    converged_tol::Integer
    # Eigen convergence
    eigentol::Real
    eigenresidual::Real
    converged_eigentol::Integer
    # Convergence checking info
    numofchecks::Integer
    converged_history::Vector{<:Integer}
    krylovbasisdim_history::Vector{<:Integer}
    maxresidual_history::Vector
    # Time needed for individual segments of the code 
    walltimes::Vector{<:Real}
    cputimes::Vector{<:Real}

    function FactorizationReport(
        convergenceinfo::ConvergenceInfo,
        factorization::KrylovFactorization,
        iterator::LanczosIterator,
        walltimes::Vector{<:Real},
        cputimes::Vector{<:Real}
    )
        # Factorization details
        factorizationtype_ = string(typeof(factorization))
        factorizationtype  = split(first(split(factorizationtype_, "{")), ".")[end]
        blocksize = isa(factorization, LanczosFactorization) ? 1 : factorization.blocksize
        basistype = typeof(factorization.basis)

        # Basis size info
        if basistype <: HybridMatrixBasis
            ncols_gpu_reserved = size(factorization.basis.gpu_basis, 2)
            ncols_cpu_reserved = size(factorization.basis.cpu_basis, 2)
            ncols_gpu_needed   = factorization.basis.nvecs_gpu
            ncols_cpu_needed   = factorization.basis.nvecs_cpu
        elseif basistype <: MatrixBasis
            reserved_cols = size(factorization.basis.basis, 2)
            needed_cols   = factorization.basis.nvecs
            if factorization.pu isa GPU
                ncols_gpu_reserved = reserved_cols
                ncols_cpu_reserved = 0
                ncols_gpu_needed   = needed_cols
                ncols_cpu_needed   = 0
            else
                ncols_cpu_reserved = reserved_cols
                ncols_gpu_reserved = 0
                ncols_cpu_needed   = needed_cols
                ncols_gpu_needed   = 0
            end
        else
            error("Unsupported basis type: $basistype")
        end

        new(
            factorizationtype,
            blocksize,
            basistype,
            ncols_cpu_reserved,
            ncols_gpu_reserved,
            ncols_cpu_needed,
            ncols_gpu_needed,
            convergenceinfo.howmany,
            convergenceinfo.numiter,
            convergenceinfo.maxiter, 
            iterator.rot,
            convergenceinfo.tol,
            convergenceinfo.residual,
            convergenceinfo.converged,
            convergenceinfo.eigentol,
            convergenceinfo.eigenresidual,
            convergenceinfo.eigenconverged,
            convergenceinfo.numofchecks,
            convergenceinfo.converged_history,
            convergenceinfo.krylovbasisdim_history,
            convergenceinfo.maxresidual_history,
            walltimes,
            cputimes
        )
    end
end



"""
    display_factorization_report(report::FactorizationReport; 
        use_colors::Bool=true, 
        show_convergence_details::Bool=false, 
        show_timings::Bool=true)

Pretty-prints a factorization report with optional ANSI colors.
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
                      remaining_percentage <= 15  ? green(f, @sprintf("%.2f%%", remaining_percentage)) :
                                                      red(f, @sprintf("%.2f%%", remaining_percentage))

    # --- Timings ---
    total_walltime = sum(report.walltimes)
    walltime, units_wt = total_walltime <= 60*2       ? (total_walltime, "seconds") :
                         total_walltime <= 60*60*2    ? (total_walltime/60, "minutes") :
                                                       (total_walltime/3600, "hours")

    total_cputime = sum(report.cputimes)
    cputime, units_ct = total_cputime <= 60*2       ? (total_cputime, "seconds") :
                        total_cputime <= 60*60*2    ? (total_cputime/60, "minutes") :
                                                      (total_cputime/3600, "hours")

    # percentages (grouping same as your code)
    perc_wt = [t / total_walltime * 100 for t in report.walltimes]
    perc_ct = [t / total_cputime * 100 for t in report.cputimes]

    perc_wt_grouped = [perc_wt[2], perc_wt[5], perc_wt[6], perc_wt[1]+perc_wt[3]+perc_wt[4]]
    perc_ct_grouped = [perc_ct[2], perc_ct[5], perc_ct[6], perc_ct[1]+perc_ct[3]+perc_ct[4]]

    formatted_perc_wt = join([cyan(f, @sprintf("%.2f%%", p)) for p in perc_wt_grouped], ", ")
    formatted_perc_ct = join([cyan(f, @sprintf("%.2f%%", p)) for p in perc_ct_grouped], ", ")

    # --- Header info ---
    fac_type = bold(f, string(report.factorizationtype))
    blocksize = bold(f, string(report.blocksize))
    howmany   = cyan(f, string(report.howmany))
    tol       = cyan(f, @sprintf("%.2e", report.tol))
    eigentol  = cyan(f, @sprintf("%.2e", report.eigentol))
    residual  = yellow(f, @sprintf("%.2e", report.residual))
    eigrnres  = yellow(f, @sprintf("%.2e", report.eigenresidual))
    basis     = cyan(f, split(first(split(string(report.basistype), "{")), ".")[end])
    rot       = cyan(f, split(string(report.rot), ".")[end])

    itersneeded   = bold(f, string(report.itersneeded))
    itersreserved = cyan(f, string(report.itersreserved))
    total_wt      = cyan(f, @sprintf("%.2f %s", walltime, units_wt))
    total_ct      = cyan(f, @sprintf("%.2f %s", cputime, units_ct))

    # --- Printing ---
    println(blue(f, "Factorization Report:"), " ($fac_type with blocksize $blocksize)")
    println("- Number of converged eigenpairs:   $eig_color (out of $howmany requested)")
    println("- Lanczos convergence satisfied by: $lan_color (with tolerance $tol, max residual $residual)")
    println("- Eigen convergence satisfied by:   $eig_color (with tolerance $eigentol, max residual $eigrnres)")
    println("- Iterations needed: $itersneeded (out of $itersreserved reserved, overestimated by $remaining_color)")
    println("- Basis type: $basis and reorthogonalization technique: $rot")

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
            header_crayon = crayon"blue bold",
            tf            = tf_unicode_rounded
        )
    end
end
