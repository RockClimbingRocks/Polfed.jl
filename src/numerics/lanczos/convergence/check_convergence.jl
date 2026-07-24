# ═══════════════════════════════════════════════════════════════════════════════
# THE ORCHESTRATOR — one function, no per-target branching.
#
# Every difference between `NumEigs`, `ToValue` and `Window` lives in `end_spec.jl`, and
# the contiguity invariant lives in `converged_run.jl`. This file only sequences them:
# cheap signal → decide → extract.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    check_convergence!(criterion::ConvergenceCriterion, state::ConvergenceState,
                       factorization::KrylovFactorization; force::Bool=false)
        -> Union{Nothing, Tuple}

Per-iteration convergence orchestrator. Returns `(vals, vecs, residuals)` when the
iteration should stop, or `nothing` to continue.

Two signals, in cost order:

1. **Ritz residuals** (cheap, [`calc_ritz_residuals`](@ref)) — computed on every scheduled
   check. They give the transformed Ritz values `p` and gate the expensive step.
2. **Eigen residuals** (`‖A·v − λ·v‖`, [`calc_eigen_residuals`](@ref)) — reconstructing
   eigenvectors is not free, so this runs only once a spec's cheap gate fires (see
   [`decide_end`](@ref)), and only for the pairs that already passed the Ritz gate.

Every returned pair is selected from a **contiguous transformed-space run** inward from the
targeted spectral extreme, in which every member passed both `tol` and `eigentol`; see
[`converged_prefix`](@ref). Mapped Window selection can trim members from that internal run,
so legacy two-sided brackets are not advertised as a contiguous energy interval.

`state.complete` records whether the stopping rule was actually met, as opposed to the run
being handed back because the budget ran out. Prefer it over re-deriving completeness from
the returned residuals.
"""
function check_convergence!(criterion::ConvergenceCriterion, state::ConvergenceState,
                            factorization::KrylovFactorization; force::Bool=false)
    state.numiter += 1
    is_final = force || state.numiter >= criterion.maxiter
    if !is_final && state.nextcheck != state.numiter
        return nothing
    end

    # Cheap signal: transformed Ritz values `p` and their residual estimates.
    p, ϕ, ritz = calc_ritz_residuals(factorization)
    ritz_ok = i -> ritz[i] < criterion.tol

    # Expensive signal, on demand: reconstruct eigenpairs for `idxs` and hand back their
    # mapped values `E` plus a per-position acceptance predicate. `decide_end` calls this
    # only after its cheap gate fires, so on most checks it is never called at all.
    eigen_of = function (idxs)
        isempty(idxs) && return (Float64[], _ -> false)
        vals, _, eig = calc_eigen_residuals(factorization, ϕ, idxs, criterion.mapvals)
        vals_cpu = is_gpu_array(vals) ? collect(vals) : vals
        return (vals_cpu, k -> eig[k] < criterion.eigentol)
    end

    # `force=is_final`: on the last check there is no next one, so the pre-gate must not
    # short-circuit the walk — the partial run has to be extracted now or it is lost.
    verdict = decide(criterion.target, p, ritz_ok, ritz, eigen_of; force=is_final)

    state.converged = length(verdict.keep)
    state.residual  = verdict.worst_residual
    state.done      = verdict.done
    state.complete  = verdict.done

    # `Inf` here just means "nothing targeted has surfaced yet" (an early check); only a
    # genuine NaN signals numerical breakdown worth warning about.
    if should_log(POLFED_WARN_LEVEL) && isnan(state.residual)
        polfed_log(
            POLFED_WARN_LEVEL,
            "Non-finite Lanczos residual detected during convergence check.",
            iteration=state.numiter, checks=state.numofchecks,
            krylovdim=factorization.krylovdim, residual=state.residual,
        )
    end

    result = nothing
    if verdict.done || is_final
        # One reconstruction of the kept set. `decide` already built these while deciding;
        # redoing it once, at the terminal check only, keeps the walker free of any
        # eigenvector cache. (`lanczos` sorts the result ascending afterwards.)
        vals, vecs, eig = calc_eigen_residuals(factorization, ϕ, verdict.keep, criterion.mapvals)
        result = (vals, vecs, eig)
        state.eigenresidual = isempty(eig) ? 0.0 : maximum(eig)

        if is_final && !verdict.done && criterion.warn_incomplete && should_log(POLFED_WARN_LEVEL)
            polfed_log(
                POLFED_WARN_LEVEL,
                "Krylov budget exhausted before the target was certified complete; " *
                "returning only selected pairs from the converged transformed prefix.",
                returned=length(verdict.keep), iteration=state.numiter,
                krylovdim=factorization.krylovdim, residual=state.residual,
            )
        end
    end

    state.numofchecks += 1
    push!(state.converged_history, state.converged)
    push!(state.krylovbasisdim_history, factorization.krylovdim)
    push!(state.maxresidual_history, state.residual)

    polfed_log(
        POLFED_DEBUG_LEVEL,
        "Convergence check evaluated.",
        iteration=state.numiter, checks=state.numofchecks,
        converged=state.converged, residual=state.residual,
        eigenresidual=state.eigenresidual, done=state.done, complete=state.complete,
        nextcheck=state.nextcheck, krylovdim=factorization.krylovdim,
    )

    state.nextcheck = criterion.next_check(state, criterion, state.numiter)
    return result
end
