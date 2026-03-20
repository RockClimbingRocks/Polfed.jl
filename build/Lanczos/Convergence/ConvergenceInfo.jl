
"""
    ConvergenceInfo(howmany, blocksize, maxiter, tol, eigentol, which, mapvals)

Mutable state tracking Lanczos convergence progress, residuals, and check
history.
"""
mutable struct ConvergenceInfo
    converged::Integer
    eigenconverged::Integer
    howmany::Integer
    residual::Real
    eigenresidual::Real
    numiter::Integer 
    maxiter::Integer
    maxdim::Integer
    nextcheck::Integer
    tol::Real
    eigentol::Real
    sorter::EigSorter
    numofchecks::Integer
    mapvals::Function 
    converged_history::Vector{<:Integer}
    krylovbasisdim_history::Vector{<:Integer}
    maxresidual_history::Vector{Float64}

    """Build initial `ConvergenceInfo` state for Lanczos iterations."""
    function ConvergenceInfo(
        howmany::Integer, 
        blocksize::Integer, 
        maxiter::Integer, 
        tol::Real, 
        eigentol::Real,
        which::Symbol, 
        mapvals::Function,
    )
        converged = 0
        eigenconverged = 0
        residual = 1.0
        eigenresidual = 1.0
        numiter = 0
        maxdim = Int(blocksize * maxiter)
        nextcheck = max(ceil(Integer, howmany ÷ blocksize + 1), 2)
        numofchecks = 0
        sorter = EigSorter(which)
        converged_history = Integer[]
        krylovbasisdim_history = Integer[]
        maxresidual_history = Float64[]
        return new(
            converged, eigenconverged, howmany, residual, eigenresidual, numiter, maxiter, maxdim, nextcheck, tol, eigentol, sorter, numofchecks, mapvals, converged_history, krylovbasisdim_history, maxresidual_history
        )
    end
end



"""
    getnumberofconvergedvecs!(convergenceinfo::ConvergenceInfo, factorization::KrylovFactorization, vecs::AbstractMatrix, vals::AbstractVector) -> nothing

Perform scheduled convergence checks and, when stopping criteria are reached,
extract/update eigenvectors, eigenvalues, and eigen-residual statistics.
"""
@inline function getnumberofconvergedvecs!(
        convergenceinfo::ConvergenceInfo, 
        factorization::KrylovFactorization,
        vecs::AbstractMatrix, 
        vals::AbstractVector
    )

    convergenceinfo.numiter += 1

    if convergenceinfo.nextcheck == convergenceinfo.numiter || convergenceinfo.numiter >= convergenceinfo.maxiter

        _, ϕ, idxs = calculate_residuals!(convergenceinfo, factorization)
        if should_log(POLFED_WARN_LEVEL) && !isfinite(convergenceinfo.residual)
            polfed_log(
                POLFED_WARN_LEVEL,
                "Non-finite Lanczos residual detected during convergence check.",
                iteration=convergenceinfo.numiter,
                checks=convergenceinfo.numofchecks,
                krylovdim=factorization.krylovdim,
                residual=convergenceinfo.residual,
            )
        end
        polfed_log(
            POLFED_DEBUG_LEVEL,
            "Convergence check evaluated.",
            iteration=convergenceinfo.numiter,
            checks=convergenceinfo.numofchecks,
            converged=convergenceinfo.converged,
            residual=convergenceinfo.residual,
            nextcheck=convergenceinfo.nextcheck,
            krylovdim=factorization.krylovdim,
        )

        if convergenceinfo.converged == convergenceinfo.howmany || convergenceinfo.numiter >= convergenceinfo.maxiter
            calculate_eigenvectors!(convergenceinfo, factorization, vecs, ϕ, idxs)
            calculate_eigenvalues!(factorization, convergenceinfo, convergenceinfo.mapvals, vecs, vals)

            calculate_eigenresiduals!(convergenceinfo, vecs, vals, factorization.pu)
            if should_log(POLFED_WARN_LEVEL)
                if !isfinite(convergenceinfo.eigenresidual)
                    polfed_log(
                        POLFED_WARN_LEVEL,
                        "Non-finite eigen residual detected after convergence extraction.",
                        iteration=convergenceinfo.numiter,
                        eigenresidual=convergenceinfo.eigenresidual,
                    )
                end
                if has_nonfinite(vals)
                    polfed_log(
                        POLFED_WARN_LEVEL,
                        "Non-finite eigenvalues detected after convergence extraction.",
                        iteration=convergenceinfo.numiter,
                    )
                end
            end
            polfed_log(
                POLFED_DEBUG_LEVEL,
                "Eigenpair extraction evaluated.",
                iteration=convergenceinfo.numiter,
                converged=convergenceinfo.converged,
                eigenconverged=convergenceinfo.eigenconverged,
                eigenresidual=convergenceinfo.eigenresidual,
            )
        end

        next_checking!(convergenceinfo, convergenceinfo.numiter, factorization)
    end
end

