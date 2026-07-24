# Default Lanczos settings, referenced by the `lanczos` entry points. Centralized
# here so the public defaults live in one place.

"""Default reorthogonalization strategy ([`FullRO`](@ref))."""
const DEFAULT_ROT = FullRO()

"""Default orthogonality-loss tolerance for the Ritz-residual convergence test."""
const DEFAULT_TOL = 1e-14

"""Default eigen-residual tolerance (`‖A·v − λ·v‖`) for accepting an eigenpair."""
const DEFAULT_EIGENTOL = 1e-10

"""
Default Krylov-dimension budget as a multiple of the number of requested
eigenpairs: `maxdim ≈ DEFAULT_MAXDIM_FACTOR · ntargets`, then **capped at the
operator dimension**. Set to the larger of the old code's two inconsistent values
so the default errs toward "run long enough to converge".

Memory cost is `dim × maxdim` for the basis, so the cap means small/medium
problems effectively run to convergence cheaply, while for large problems you
should pass `maxdim` explicitly to bound memory (POLFED always does).
"""
const DEFAULT_MAXDIM_FACTOR = 500

"""
    default_next_check(state::ConvergenceState, criterion::ConvergenceCriterion, k::Int) -> Int

Default convergence-check scheduling function. Returns the next iteration at which a
full projected-matrix eigendecomposition + residual scan should run, adapting the
interval to the current Ritz residual `state.residual`:

| current residual `ε` | next check after |
|----------------------|------------------|
| `ε > 1e-2`           | `k + 20`         |
| `1e-3 < ε ≤ 1e-2`    | `k + 10`         |
| `ε ≤ 1e-3`           | `k + 5`          |

Checks are cheap to schedule rarely when far from convergence and more often as the
residual shrinks. Pass a custom function as `next_check` to [`ConvergenceCriterion`](@ref)
to override this policy — the callback receives `state` (current counters and history)
and `criterion` (budget: `maxdim`, `maxiter`, `blocksize`) so any adaptive strategy
is expressible.
"""
function default_next_check(state::ConvergenceState, ::ConvergenceCriterion, k::Int)
    ε = state.residual
    if     ε > 1e-2;  return k + 20
    elseif ε > 1e-3;  return k + 10
    else              return k + 5
    end
end

"""Default convergence-check scheduling callback ([`default_next_check`](@ref))."""
const DEFAULT_NEXT_CHECK = default_next_check
