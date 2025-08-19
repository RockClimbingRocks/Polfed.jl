
include("EigSorter.jl")

mutable struct ConvergenceInfo
    converged::Int
    howmany::Int
    residual::Real
    numiter::Int 
    maxiter::Int
    nextcheck::Int
    tol::Real
    sorter::EigSorter
    numofchecks::Int
    converged_history::Vector{<:Int}
    krylovbasisdim_history::Vector{<:Int}
    maxresidual_history::Vector{Float64}


    function ConvergenceInfo(howmany::Int, s::Int, maxiter::Int, tol::Real, sorter::EigSorter)
        converged = 0
        residual = 1.
        numiter = 0
        nextcheck = max(ceil(Int, howmany÷s+1),2)
        numofchecks = 0
        converged_history = Vector{Int}()
        krylovbasisdim_history = Vector{Int}()
        maxresidual_history = Vector{Float64}()

        return new(converged, howmany, residual, numiter, maxiter, nextcheck, tol, sorter, numofchecks, converged_history, krylovbasisdim_history, maxresidual_history)
    end
end

mutable struct EigenvaluesInfo
    tol::Real
    mapvals::Function
    eigenresidual::Real
    converged::Int

    function EigenvaluesInfo(tol::Real, mapvals::Function)
        converged = 0
        eigenresidual = 1.
        return new(tol, mapvals, eigenresidual, converged)
    end
end

mutable struct ConvergenceInfoOut
    # Factorization info
    factorizationtype::String
    blocksize::Int
    # Requested number of eigrnpairs
    howmany::Int
    # Lanczos convergence
    tol::Real
    residual::Real
    converged_tol::Int
    # Eigen convergence
    eigentol::Real
    eigrnresidual::Real
    converged_eigentol::Int
    # Preallocation/Estimation of memory
    itersneeded::Int
    itersreserved::Int
    # Convergence checking info
    numofchecks::Int
    converged_history::Vector{<:Integer}
    krylovbasisdim_history::Vector{<:Integer}
    maxresidual_history::Vector
    # Time needed for individual segments of the code 
    walltimes::Vector{<:Real}


    # ConvergenceInfoOut(::QSystem.Lanczos.ConvergenceInfo, ::QSystem.Lanczos.EigenvaluesInfo, ::QSystem.Lanczos.BlockLanczosFactorization{Float64, QSystem.Lanczos.GPU}, ::Vector{Float64})
    function ConvergenceInfoOut(convergenceinfo::ConvergenceInfo, eigenvaluesinfo::EigenvaluesInfo, factorization::KrylovFactorization, walltimes::Vector{Float64})
        factorizationtype_ = string(typeof(factorization))
        factorizationtype = split(first(split(factorizationtype_,"{")), ".")[end]
        blocksize = isa(factorization, LanczosFactorization) ? 1 : factorization.blocksize
        new(factorizationtype, 
            blocksize,
            convergenceinfo.howmany, 
            convergenceinfo.tol, 
            convergenceinfo.residual, 
            convergenceinfo.converged, 
            eigenvaluesinfo.tol, 
            eigenvaluesinfo.eigenresidual, 
            eigenvaluesinfo.converged, 
            convergenceinfo.numiter, 
            convergenceinfo.maxiter, 
            convergenceinfo.numofchecks,
            convergenceinfo.converged_history,
            convergenceinfo.krylovbasisdim_history,
            convergenceinfo.maxresidual_history,
            walltimes,
        )
    end


end

include("getvecs.jl")
include("getvals.jl")
include("CalculatingResiduals.jl")
include("NextChecking.jl")


@inline function getnumberofconvergedvecs!(
        convergenceinfo::ConvergenceInfo, 
        factorization::KrylovFactorization, 
        valsinfo::EigenvaluesInfo, 
        vecs::AbstractMatrix, 
        vals::AbstractVector
    )

    convergenceinfo.numiter += 1

    if convergenceinfo.nextcheck == convergenceinfo.numiter || convergenceinfo.numiter >= convergenceinfo.maxiter

        _, ϕ, idxs = calculate_residuals!(convergenceinfo, factorization)

        if convergenceinfo.converged == convergenceinfo.howmany || convergenceinfo.numiter >= convergenceinfo.maxiter
            calculate_eigenvectors!(convergenceinfo, factorization, vecs, ϕ, idxs)
            calculate_eigenvalues!(factorization, convergenceinfo, valsinfo.mapvals, vecs, vals)

            calculate_eigenresiduals!(convergenceinfo, valsinfo, vecs, vals, factorization.pu)
        end

        next_checking!(convergenceinfo, convergenceinfo.numiter, factorization)
    end
end






function display_convergenceinfo(info::ConvergenceInfoOut; show_convergence_details=false)


    eig_tol_color = info.converged_eigentol == info.howmany ? "\e[1;32m" : "\e[1;31m"
    lanczos_tol_color = info.converged_tol == info.howmany ? "\e[1;32m" : "\e[1;31m"
    remaining_percentage = (1 - info.itersneeded / info.itersreserved) * 100
    remaining_color = remaining_percentage <= 3 ? "\e[1;31m" :
                      remaining_percentage <= 5 ? "\e[1;33m" :
                      remaining_percentage <= 15 ? "\e[1;32m" : "\e[1;31m"

    total_walltime = sum(info.walltimes) 
    walltime, units =   total_walltime <= 60*2 ? (total_walltime, "seconds") :
                        total_walltime <= 60*60*2 ? (total_walltime/60, "minutes") : (total_walltime/60/60, "hours")
    

    percentages = [t / total_walltime * 100 for t in info.walltimes]
    formatted_percentages = join(["\e[1;36m" * @sprintf("%.2f", p) * "%\e[0m" for p in percentages], ", ")

    factorizationtype   = @sprintf("\e[1m%s\e[0m", info.factorizationtype)
    blocksize           = @sprintf("\e[1m%d\e[0m", info.blocksize)
    howmany             = @sprintf("\e[1;36m %d \e[0m", info.howmany)
    converged           = @sprintf("%s %d \e[0m", eig_tol_color, info.converged_eigentol) 
    lanczos_coverged    = @sprintf("%s %d \e[0m", lanczos_tol_color, info.converged_tol)
    eigen_coverged      = @sprintf("%s %d \e[0m", eig_tol_color, info.converged_eigentol) 
    tol                 = @sprintf("\e[1;36m %.2e \e[0m", info.tol)
    eigentol            = @sprintf("\e[1;36m %.2e \e[0m", info.eigentol)
    residual            = @sprintf("\e[1;33m %.2e \e[0m", info.residual)
    eigrnresidual       = @sprintf("\e[1;33m %.2e \e[0m", info.eigrnresidual)

    itersneeded         = @sprintf("%s %d \e[0m", remaining_color, info.itersneeded)
    itersreserved       = @sprintf("\e[1;36m %d \e[0m", info.itersreserved)
    iterspercantage     = @sprintf("%s %.2f%% \e[0m", remaining_color, remaining_percentage) 
    totaltime           = @sprintf("\e[1;36m %.2f %s \e[0m", walltime, units)

    numofchecks         = @sprintf("\e[1m%d\e[0m", info.numofchecks)

    print(  "\e[1;34mConvergence Information:\e[0m")
    isa(factorizationtype, LanczosFactorization) ? println(  " ($factorizationtype)") : println(" ($factorizationtype with blocksize $blocksize)")
    println("- Number of converged eigenpairs:   $converged (out of ", howmany, " requested)")
    println("- Lanczos convergence satisfied by: $lanczos_coverged (within tolerance $tol maximal residual was $residual)")
    println("- Eigen convergence satisfied by:   $eigen_coverged (within tolerance $eigentol maximal residual was $eigrnresidual)")
    println("- Iterations needed: $itersneeded  (out of $itersreserved reserved, overestimated by $iterspercantage)")
    println("- Lanczos method took: $totaltime (distributed as: $formatted_percentages )")

    if show_convergence_details

        header = (["Checking", "Krylov dim.", "Converged", "Residual"])
        data = hcat(1:info.numofchecks, info.krylovbasisdim_history, info.converged_history, info.maxresidual_history);


        println("- Convergence check was peerformed $(info.numofchecks) times, here is the table of results:")

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




function print_convergenceinfo(info::ConvergenceInfoOut; show_convergence_details=false)


    eig_tol_color = info.converged_eigentol == info.howmany ? "" : ""
    lanczos_tol_color = info.converged_tol == info.howmany ? "" : ""
    remaining_percentage = (1 - info.itersneeded / info.itersreserved) * 100
    remaining_color = remaining_percentage <= 3 ? "" :
                      remaining_percentage <= 5 ? "" :
                      remaining_percentage <= 15 ? "" : ""

    total_walltime = sum(info.walltimes) 
    walltime, units =   total_walltime <= 60*2 ? (total_walltime, "seconds") :
                        total_walltime <= 60*60*2 ? (total_walltime/60, "minutes") : (total_walltime/60/60, "hours")
    

    percentages = [t / total_walltime * 100 for t in info.walltimes]
    formatted_percentages = join(["" * @sprintf("%.2f", p) * "" for p in percentages], ", ")

    factorizationtype   = @sprintf("%s", info.factorizationtype)
    blocksize           = @sprintf("%d", info.blocksize)
    howmany             = @sprintf(" %d ", info.howmany)
    converged           = @sprintf("%s %d ", eig_tol_color, info.converged_eigentol) 
    lanczos_coverged    = @sprintf("%s %d ", lanczos_tol_color, info.converged_tol)
    eigen_coverged      = @sprintf("%s %d ", eig_tol_color, info.converged_eigentol) 
    tol                 = @sprintf(" %.2e ", info.tol)
    eigentol            = @sprintf(" %.2e ", info.eigentol)
    residual            = @sprintf(" %.2e ", info.residual)
    eigrnresidual       = @sprintf(" %.2e ", info.eigrnresidual)

    itersneeded         = @sprintf("%s %d ", remaining_color, info.itersneeded)
    itersreserved       = @sprintf(" %d ", info.itersreserved)
    iterspercantage     = @sprintf("%s %.2f%% ", remaining_color, remaining_percentage) 
    totaltime           = @sprintf(" %.2f %s ", walltime, units)

    numofchecks         = @sprintf("%d", info.numofchecks)

    print(  "Convergence Information:")
    isa(factorizationtype, LanczosFactorization) ? println(  " ($factorizationtype)") : println(" ($factorizationtype with blocksize $blocksize)")
    println("- Number of converged eigenpairs:   $converged (out of ", howmany, " requested)")
    println("- Lanczos convergence satisfied by: $lanczos_coverged (within tolerance $tol maximal residual was $residual)")
    println("- Eigen convergence satisfied by:   $eigen_coverged (within tolerance $eigentol maximal residual was $eigrnresidual)")
    println("- Iterations needed: $itersneeded  (out of $itersreserved reserved, overestimated by $iterspercantage)")
    println("- Lanczos method took: $totaltime (distributed as: $formatted_percentages )")

    if show_convergence_details

        header = (["Checking", "Krylov dim.", "Converged", "Residual"])
        data = hcat(1:info.numofchecks, info.krylovbasisdim_history, info.converged_history, info.maxresidual_history);


        println("- Convergence check was peerformed $(info.numofchecks) times, here is the table of results:")

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