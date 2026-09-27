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

`state.ritzconverged` / `state.eigenconverged` count how many pairs pass each gate on its own.
When the Ritz count exceeds `state.converged`, the walk was truncated by `eigentol` — a
condition no amount of extra iterations fixes, and one the Ritz residual alone cannot show (it
is at its floor precisely because the Krylov space converged). When the EIGEN count also
exceeds it, the contiguity invariant is discarding pairs that passed both gates but sit behind
a failure.
"""
function check_convergence!(criterion::ConvergenceCriterion, state::ConvergenceState,
                            factorization::KrylovFactorization; force::Bool=false)
    state.numiter += 1
    is_final = force || state.numiter >= criterion.maxiter
    if !is_final && state.nextcheck != state.numiter
        return nothing
    end

    # Wall clock for THIS check only, started past the early return so the timing covers work
    # that actually happened. The ambient `:conv_check` stage accumulates every call including
    # the skipped ones, which is the right total but cannot show a per-check cost.
    t_start = time()

    # Cheap signal: transformed Ritz values `p` and their residual estimates.
    p, ϕ, ritz = calc_ritz_residuals(factorization)
    ritz_ok = i -> ritz[i] < criterion.tol

    # Expensive signal, on demand: the mapped values `E` of `idxs` plus a per-position
    # acceptance predicate. `decide_end` calls this only after its cheap gate fires, so on
    # most checks it is never called at all. The eigenvectors are rebuilt a chunk at a time
    # and dropped: the iteration may continue, so the basis must survive, and keeping them
    # would put a full eigenvector matrix on top of it.
    eigen_of = function (idxs)
        isempty(idxs) && return (Float64[], _ -> false, Float64[])
        vals, eig = calc_eigen_residuals_chunked(factorization, ϕ, idxs, criterion.mapvals)
        return (vals, k -> eig[k] < criterion.eigentol, eig)
    end

    # `force=is_final`: on the last check there is no next one, so the pre-gate must not
    # short-circuit the walk — the partial run has to be extracted now or it is lost.
    verdict = decide(criterion.target, p, ritz_ok, ritz, eigen_of; force=is_final)

    state.converged      = length(verdict.keep)
    state.ritzconverged  = verdict.n_ritz
    state.eigenconverged = verdict.n_eigen
    state.eigenworst     = verdict.eigen_worst
    state.residual      = verdict.worst_residual
    state.done          = verdict.done
    state.complete      = verdict.done

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
        # The iteration stops here, so the kept eigenvectors are built inside the Krylov basis
        # instead of next to it: the peak stays at the basis. (`lanczos` sorts the result
        # ascending afterwards, also in place.)
        vals, vecs, eig = extract_eigenpairs!(factorization, ϕ, verdict.keep, criterion.mapvals)
        result = (vals, vecs, eig)
        state.eigenresidual = isempty(eig) ? 0.0 : maximum(eig)

        if is_final && !verdict.done && criterion.warn_incomplete && should_log(POLFED_WARN_LEVEL)
            # Which gate actually stopped the walk decides the advice. When the Ritz prefix
            # reached further than the returned run, the Krylov space did its job and
            # `eigentol` is the binding constraint — more iterations cannot help, and telling
            # the caller to widen the budget sends them the wrong way.
            if verdict.n_ritz > length(verdict.keep)
                polfed_log(
                    POLFED_WARN_LEVEL,
                    "Stopped by the EIGEN tolerance, not the Krylov budget: the walk passed " *
                    "the Ritz gate but was cut where ‖A·v − λ·v‖ exceeded `eigentol`. More " *
                    "iterations will not extend it — relax `eigentol`, or narrow the target " *
                    "window so the filter separates neighbouring states more strongly.",
                    returned=length(verdict.keep), passed_ritz=verdict.n_ritz,
                    passed_eigen=verdict.n_eigen, eigentol=criterion.eigentol,
                    worst_eigen_residual=verdict.eigen_worst,
                    iteration=state.numiter, krylovdim=factorization.krylovdim,
                    ritz_residual=state.residual,
                )
            else
                polfed_log(
                    POLFED_WARN_LEVEL,
                    "Krylov budget exhausted before the target was certified complete; " *
                    "returning only selected pairs from the converged transformed prefix.",
                    returned=length(verdict.keep), iteration=state.numiter,
                    krylovdim=factorization.krylovdim, residual=state.residual,
                )
            end
        end
    end

    state.numofchecks += 1
    push!(state.converged_history, state.converged)
    push!(state.ritzconverged_history, state.ritzconverged)
    push!(state.eigenconverged_history, state.eigenconverged)
    push!(state.krylovbasisdim_history, factorization.krylovdim)
    push!(state.maxresidual_history, state.residual)
    push!(state.eigenresidual_history, state.eigenworst)
    push!(state.checktime_history, time() - t_start)

    polfed_log(
        POLFED_DEBUG_LEVEL,
        "Convergence check evaluated.",
        iteration=state.numiter, checks=state.numofchecks,
        converged=state.converged, passed_ritz=state.ritzconverged,
        passed_eigen=state.eigenconverged,
        residual=state.residual, worst_eigen=state.eigenworst,
        eigenresidual=state.eigenresidual, done=state.done, complete=state.complete,
        nextcheck=state.nextcheck, krylovdim=factorization.krylovdim,
    )

    state.nextcheck = criterion.next_check(state, criterion, state.numiter)
    return result
end
