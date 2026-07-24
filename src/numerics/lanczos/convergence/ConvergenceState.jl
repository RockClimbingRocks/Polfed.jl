"""
    ConvergenceState

Mutable Lanczos convergence state (running progress), with concrete field types.
Paired with the immutable [`ConvergenceCriterion`](@ref).

# Fields
- `converged::Int`: number of currently selected pairs. Each comes from the internal
  transformed-space prefix and passed both the Ritz (`tol`) and eigen (`eigentol`) gates.
- `done::Bool`: the target's stopping rule is satisfied.
- `complete::Bool`: the returned set is **certified complete** by the criterion — the
  stopping rule was met, as opposed to the run merely being handed back because the Krylov
  budget ran out. Consumers should prefer this over re-deriving completeness from
  residuals downstream.
- `residual::Float64`: Ritz progress signal consumed by the next-check scheduler. It is the
  first required blocking residual while incomplete and the worst required residual once
  complete.
- `eigenresidual::Float64`: worst eigen residual over the extracted run.
- `numiter::Int`, `nextcheck::Int`, `numofchecks::Int`: iteration / scheduling counters.
- `converged_history`, `krylovbasisdim_history`, `maxresidual_history`: per-check history.
"""
mutable struct ConvergenceState
    converged::Int
    done::Bool
    complete::Bool
    residual::Float64
    eigenresidual::Float64
    numiter::Int
    nextcheck::Int
    numofchecks::Int
    converged_history::Vector{Int}
    krylovbasisdim_history::Vector{Int}
    maxresidual_history::Vector{Float64}
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
    return ConvergenceState(0, false, false, 1.0, 1.0, 0, nextcheck, 0, Int[], Int[], Float64[])
end
