"""
    FactorizationReport(convergenceinfo::ConvergenceInfo,
                        factorization::KrylovFactorization,
                        iterator::LanczosIterator,
                        walltimes::Vector{<:Real},
                        cputimes::Vector{<:Real})

Stores detailed information about a single factorization run, including basis statistics, convergence results, and timing data.

This report is typically constructed internally after a Lanczos or block-Lanczos factorization completes, and it provides the core diagnostic information used by higher-level reporting tools.

# Arguments
- `convergenceinfo::ConvergenceInfo`: 
  Summary of convergence behavior, including tolerances, number of converged eigenpairs, iteration counts, and convergence history.
- `factorization::KrylovFactorization`: 
  The factorization object (e.g., `LanczosFactorization` or `BlockLanczosFactorization`) containing the computed Krylov basis and internal parameters.
- `iterator::LanczosIterator`: 
  The iterator used to build the Krylov subspace, providing information about reorthogonalization techniques and iteration limits.
- `walltimes::Vector{<:Real}`: 
  Wall-clock times (in seconds) for each major code segment of the factorization routine.
- `cputimes::Vector{<:Real}`: 
  CPU times (in seconds) corresponding to the same code segments as in `walltimes`.

# Fields
- `factorizationtype::String`: Type of the factorization method used (e.g., `"LanczosFactorization"`).
- `blocksize::Integer`: Block size used in the factorization (1 for standard Lanczos).
- `basistype::Type{<:OrthonormalBasis}`: Type of orthonormal basis used (e.g., `MatrixBasis`, `HybridMatrixBasis`).
- `ncols_cpu_reserved`, `ncols_gpu_reserved::Integer`: Number of basis vectors reserved on CPU and GPU, respectively.
- `ncols_cpu_needed`, `ncols_gpu_needed::Integer`: Number of basis vectors actually used during the run.
- `howmany::Integer`: Number of eigenpairs requested.
- `itersneeded::Integer`: Number of iterations performed before convergence.
- `itersreserved::Integer`: Number of iterations preallocated or reserved.
- `rot::ReOrthTechnique`: Reorthogonalization method used (e.g., `FullReorth`, `PartialReorth`).
- `tol`, `residual::Real`: Lanczos convergence tolerance and final residual norm.
- `converged_tol::Integer`: Number of eigenpairs satisfying the Lanczos convergence tolerance.
- `eigentol`, `eigenresidual::Real`: Eigenvalue convergence tolerance and corresponding residual.
- `converged_eigentol::Integer`: Number of eigenpairs satisfying the eigenvalue convergence tolerance.
- `numofchecks::Integer`: Number of convergence checks performed during the run.
- `converged_history::Vector{<:Integer}`: Number of converged eigenpairs at each check.
- `krylovbasisdim_history::Vector{<:Integer}`: Krylov subspace dimension recorded at each check.
- `maxresidual_history::Vector`: Maximum residual per convergence check.
- `walltimes`, `cputimes::Vector{<:Real}`: Timings for individual stages (e.g., mapping, reorthogonalization, convergence check).

# Notes
- This report is primarily used by [`display_factorization_report`](@ref) to summarize performance and convergence behavior.
- Supports both CPU-only and hybrid CPU/GPU basis types.
"""
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

    """Build `FactorizationReport` from convergence/factorization runtime state."""
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

Displays a detailed summary of a `FactorizationReport` object, including convergence, iteration, and timing information.  
Optionally includes a convergence history table for fine-grained inspection.

# Arguments
- `report::FactorizationReport`: The report object to display.
- `use_colors::Bool=true`: If `true`, enables ANSI color formatting for terminal readability.
- `show_convergence_details::Bool=false`: If `true`, prints a formatted table showing convergence evolution across checks.
- `show_timings::Bool=true`: If `true`, includes timing summaries for walltime and CPU time, including stage-wise percentages.

# Output
The function prints:
1. **Convergence summary** – how many eigenpairs reached the desired tolerance.
2. **Iteration details** – total iterations used vs. reserved.
3. **Basis information** – type of basis and reorthogonalization strategy.
4. **Timing summary** – breakdown of walltime and CPU time.
5. (Optional) **Convergence table** – evolution of convergence metrics if `show_convergence_details=true`.

# Notes
- Uses color highlighting to indicate convergence success:
  - ✅ Green → all requested eigenpairs converged.
  - ⚠️ Yellow → moderate margin to reserved iterations.
  - ❌ Red → insufficient convergence or heavy overestimation.
- Timing percentages are grouped as:
  `(Mapping, Reorthogonalization, Convergence check, others)`
- To suppress colors for non-TTY environments (e.g., when redirecting output to file), set `use_colors=false`.
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
