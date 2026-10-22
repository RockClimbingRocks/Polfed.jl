"""
    ConvergenceState

Mutable Lanczos convergence state (running progress), with concrete field types.
Paired with the immutable [`ConvergenceCriterion`](@ref).

# Fields
- `converged::Int`: number of currently selected pairs. Each comes from the internal
  transformed-space prefix and passed both the Ritz (`tol`) and eigen (`eigentol`) gates.
- `ritzconverged::Int` / `eigenconverged::Int`: how many pairs pass each gate on its own —
  plain counts, unlike `converged`, which is a contiguous run. `eigenconverged == -1` means the
  cheap pre-gate skipped reconstruction, so nothing was measured. Two readings fall out of
  comparing them with `converged`: `ritzconverged > converged` says the eigen tolerance (not the
  Krylov budget) truncated the answer — the one diagnosis the Ritz residual cannot give, since
  it sits at its floor exactly when the Krylov space has converged — and `eigenconverged >
  converged` says the contiguity invariant then dropped pairs that had passed both gates,
  because a failure sits between them and the extreme.
- `done::Bool`: the target's stopping rule is satisfied.
- `complete::Bool`: the returned set is **certified complete** by the criterion — the
  stopping rule was met, as opposed to the run merely being handed back because the Krylov
  budget ran out. Consumers should prefer this over re-deriving completeness from
  residuals downstream.
- `residual::Float64`: Ritz progress signal consumed by the next-check scheduler. It is the
  first required blocking residual while incomplete and the worst required residual once
  complete.
- `eigenresidual::Float64`: worst eigen residual over the extracted run (terminal check only).
- `eigenworst::Float64`: worst eigen residual over everything *examined* at the last check —
  including the pairs that failed. `NaN` when the cheap pre-gate skipped the reconstruction.
  This, next to `residual`, is what makes an eigen-gated stall legible.
- `numiter::Int`, `nextcheck::Int`, `numofchecks::Int`: iteration / scheduling counters.
- `converged_history`, `ritzconverged_history`, `eigenconverged_history`,
  `krylovbasisdim_history`, `maxresidual_history`, `eigenresidual_history`,
  `checktime_history`: per-check history. `checktime_history` is the WALL time each check that
  actually ran took (skipped iterations record nothing) — the reconstruction is the expensive
  half of a POLFED run, so the per-check cost belongs next to the per-check progress.
"""
mutable struct ConvergenceState
    converged::Int
    ritzconverged::Int
    eigenconverged::Int
    done::Bool
    complete::Bool
    residual::Float64
    eigenresidual::Float64
    eigenworst::Float64
    numiter::Int
    nextcheck::Int
    numofchecks::Int
    converged_history::Vector{Int}
    ritzconverged_history::Vector{Int}
    eigenconverged_history::Vector{Int}
    krylovbasisdim_history::Vector{Int}
    maxresidual_history::Vector{Float64}
    eigenresidual_history::Vector{Float64}
    checktime_history::Vector{Float64}
end

"""
    ConvergenceState(criterion::ConvergenceCriterion) -> ConvergenceState

Build the initial state. The first check is scheduled once the Krylov subspace can
hold the requested count (or early, for value-based targets whose count is dynamic).
"""
function ConvergenceState(criterion::ConvergenceCriterion)
    rc = requested_count(criterion.target)
    base = rc > 0 ? rc : 1
    nextcheck = max(cld(base, criterion.blocksize) + 1, 2)
    return ConvergenceState(0, 0, -1, false, false, 1.0, 1.0, NaN, 0, nextcheck, 0,
                            Int[], Int[], Int[], Int[], Float64[], Float64[], Float64[])
end
