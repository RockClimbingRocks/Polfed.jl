mutable struct FactorizationReport
    # Factorization info
    factorizationtype::String
    blockdim::Integer
    # Basis related stuff
    basistype::Type{<:OrthonormalBasis}
    ncols_cpu_reserved::Integer
    ncols_gpu_reserved::Integer
    ncols_cpu_needed::Integer
    ncols_gpu_needed::Integer
    howmany::Integer
    itersneeded::Integer
    itersreserved::Integer
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
        walltimes::Vector{<:Real},
        cputimes::Vector{<:Real}
    )
        # Factorization details
        factorizationtype_ = string(typeof(factorization))
        factorizationtype  = split(first(split(factorizationtype_, "{")), ".")[end]
        blockdim = isa(factorization, LanczosFactorization) ? 1 : factorization.blockdim
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
            blockdim,
            basistype,
            ncols_cpu_reserved,
            ncols_gpu_reserved,
            ncols_cpu_needed,
            ncols_gpu_needed,
            convergenceinfo.howmany,
            convergenceinfo.numiter,          # itersneeded
            convergenceinfo.maxiter,          # itersreserved
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


function display_report(report::FactorizationReport; show_convergence_details=false, show_timings=true)


    eig_tol_color = report.converged_eigentol == report.howmany ? "\e[1;32m" : "\e[1;31m"
    lanczos_tol_color = report.converged_tol == report.howmany ? "\e[1;32m" : "\e[1;31m"
    remaining_percentage = (1 - report.itersneeded / report.itersreserved) * 100
    remaining_color = remaining_percentage <= 3 ? "\e[1;31m" :
                      remaining_percentage <= 5 ? "\e[1;33m" :
                      remaining_percentage <= 15 ? "\e[1;32m" : "\e[1;31m"

    total_walltime = sum(report.walltimes) 
    walltime, units_wt =   total_walltime <= 60*2 ? (total_walltime, "seconds") :
                        total_walltime <= 60*60*2 ? (total_walltime/60, "minutes") : (total_walltime/60/60, "hours")
    percentages_wt = [t / total_walltime * 100 for t in report.walltimes]
    percentages_ordered_grouped_wt = [percentages_wt[2], percentages_wt[5], percentages_wt[6], percentages_wt[1]+percentages_wt[3]+percentages_wt[4]]
    formatted_percentages_wt = join(["\e[1;36m" * @sprintf("%.2f", p) * "%\e[0m" for p in percentages_ordered_grouped_wt], ", ")



    total_cputime = sum(report.cputimes) 
    cputime, units_ct =   total_cputime <= 60*2 ? (total_cputime, "seconds") :
                        total_cputime <= 60*60*2 ? (total_cputime/60, "minutes") : (total_cputime/60/60, "hours")
    percentages_ct = [t / total_cputime * 100 for t in report.cputimes]
    percentages_ordered_grouped_ct = [percentages_ct[2], percentages_ct[5], percentages_ct[6], percentages_ct[1]+percentages_ct[3]+percentages_ct[4]]
    formatted_percentages_ct = join(["\e[1;36m" * @sprintf("%.2f", p) * "%\e[0m" for p in percentages_ordered_grouped_ct], ", ")



    factorizationtype   = @sprintf("\e[1m%s\e[0m", report.factorizationtype)
    blocksize           = @sprintf("\e[1m%d\e[0m", report.blockdim)
    howmany             = @sprintf("\e[1;36m %d \e[0m", report.howmany)
    converged           = @sprintf("%s %d \e[0m", eig_tol_color, report.converged_eigentol) 
    lanczos_coverged    = @sprintf("%s %d \e[0m", lanczos_tol_color, report.converged_tol)
    eigen_coverged      = @sprintf("%s %d \e[0m", eig_tol_color, report.converged_eigentol) 
    tol                 = @sprintf("\e[1;36m %.2e \e[0m", report.tol)
    eigentol            = @sprintf("\e[1;36m %.2e \e[0m", report.eigentol)
    residual            = @sprintf("\e[1;33m %.2e \e[0m", report.residual)
    eigrnresidual       = @sprintf("\e[1;33m %.2e \e[0m", report.eigenresidual)

    itersneeded         = @sprintf("%s %d \e[0m", remaining_color, report.itersneeded)
    itersreserved       = @sprintf("\e[1;36m %d \e[0m", report.itersreserved)
    iterspercantage     = @sprintf("%s %.2f%% \e[0m", remaining_color, remaining_percentage) 
    total_wt            = @sprintf("\e[1;36m %.2f %s \e[0m", walltime, units_wt)
    total_ct            = @sprintf("\e[1;36m %.2f %s \e[0m", cputime, units_ct)

    numofchecks         = @sprintf("\e[1m%d\e[0m", report.numofchecks)

    print(  "\e[1;34mFactorization Report:\e[0m")
    isa(factorizationtype, LanczosFactorization) ? println(  " ($factorizationtype)") : println(" ($factorizationtype with blocksize $blocksize)")
    println("- Number of converged eigenpairs:   $converged (out of ", howmany, " requested)")
    println("- Lanczos convergence satisfied by: $lanczos_coverged (with tolerance $tol max residual was $residual)")
    println("- Eigen convergence satisfied by:   $eigen_coverged (with tolerance $eigentol max residual was $eigrnresidual)")
    println("- Iterations needed: $itersneeded  (out of $itersreserved reserved, overestimated by $iterspercantage)")

    if show_timings
        println(  "\e[1;34mTimings:\e[0m Percentages are distributed as: (Mapping, Reorthogonalization, Convergence check, others)")
        println("- Walltime of factorization took: $total_wt ($formatted_percentages_wt)")
        println("- CPU time of factorization took: $total_ct ($formatted_percentages_ct)")
    end 

    if show_convergence_details

        header = (["Checking", "Krylov dim.", "Converged", "Residual"])
        data = hcat(1:report.numofchecks, report.krylovbasisdim_history, report.converged_history, report.maxresidual_history);


        println("- Convergence check was peerformed $(report.numofchecks) times, here is the table of results:")

        pretty_table(
               data;
               formatters    = ft_printf("%d", 1:3),
               header        = header,
               header_crayon = crayon"blue bold",
               tf            = tf_unicode_rounded
               # formatters    = ft_printf("%5å.2f", 2:4),
               # highlighters  = (hl_10, hl_p, hl_v),
           )
    end
end


function print_report(report::FactorizationReport; show_convergence_details=false)


    eig_tol_color = report.converged_eigentol == report.howmany ? "" : ""
    lanczos_tol_color = report.converged_tol == report.howmany ? "" : ""
    remaining_percentage = (1 - report.itersneeded / report.itersreserved) * 100
    remaining_color = remaining_percentage <= 3 ? "" :
                      remaining_percentage <= 5 ? "" :
                      remaining_percentage <= 15 ? "" : ""

    total_walltime = sum(report.walltimes) 
    walltime, units =   total_walltime <= 60*2 ? (total_walltime, "seconds") :
                        total_walltime <= 60*60*2 ? (total_walltime/60, "minutes") : (total_walltime/60/60, "hours")
    

    percentages = [t / total_walltime * 100 for t in report.walltimes]
    formatted_percentages = join(["" * @sprintf("%.2f", p) * "" for p in percentages], ", ")

    factorizationtype   = @sprintf("%s", report.factorizationtype)
    blocksize           = @sprintf("%d", report.blockdim)
    howmany             = @sprintf(" %d ", report.howmany)
    converged           = @sprintf("%s %d ", eig_tol_color, report.converged_eigentol) 
    lanczos_coverged    = @sprintf("%s %d ", lanczos_tol_color, report.converged_tol)
    eigen_coverged      = @sprintf("%s %d ", eig_tol_color, report.converged_eigentol) 
    tol                 = @sprintf(" %.2e ", report.tol)
    eigentol            = @sprintf(" %.2e ", report.eigentol)
    residual            = @sprintf(" %.2e ", report.residual)
    eigrnresidual       = @sprintf(" %.2e ", report.eigrnresidual)

    itersneeded         = @sprintf("%s %d ", remaining_color, report.itersneeded)
    itersreserved       = @sprintf(" %d ", report.itersreserved)
    iterspercantage     = @sprintf("%s %.2f%% ", remaining_color, remaining_percentage) 
    totaltime           = @sprintf(" %.2f %s ", walltime, units)

    numofchecks         = @sprintf("%d", report.numofchecks)

    print(  "Convergence Information:")
    isa(factorizationtype, LanczosFactorization) ? println(  " ($factorizationtype)") : println(" ($factorizationtype with blocksize $blocksize)")
    println("- Number of converged eigenpairs:   $converged (out of ", howmany, " requested)")
    println("- Lanczos convergence satisfied by: $lanczos_coverged (within tolerance $tol maximal residual was $residual)")
    println("- Eigen convergence satisfied by:   $eigen_coverged (within tolerance $eigentol maximal residual was $eigrnresidual)")
    println("- Iterations needed: $itersneeded  (out of $itersreserved reserved, overestimated by $iterspercantage)")
    println("- Lanczos method took: $totaltime (distributed as: $formatted_percentages )")

    if show_convergence_details

        header = (["Checking", "Krylov dim.", "Converged", "Residual"])
        data = hcat(1:report.numofchecks, report.krylovbasisdim_history, report.converged_history, report.maxresidual_history);


        println("- Convergence check was peerformed $(report.numofchecks) times, here is the table of results:")

        pretty_table(
               data;
               formatters    = ft_printf("%d", 1:3),
               header        = header,
               header_crayon = crayon"blue bold",
               tf            = tf_unicode_rounded
               # formatters    = ft_printf("%5å.2f", 2:4),
               # highlighters  = (hl_10, hl_p, hl_v),
           )
    end
end
