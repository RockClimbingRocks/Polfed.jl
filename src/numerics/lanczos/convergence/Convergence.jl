# Convergence subsystem: targeting (EigTarget), config/state split
# (ConvergenceCriterion / ConvergenceState), residual evaluation, eigenpair
# extraction, scheduling, and the per-iteration orchestrator.
#
# Layered bottom-up, and the layering is the design:
#   EigTarget.jl      types ONLY — no logic
#   end_spec.jl       the per-spec interface (the whole "what does convergence
#                     mean" surface; adding a spec means adding a section there)
#   converged_run.jl  the walker — enforces a no-skip transformed-space prefix before
#                     mapped selection
#   check_convergence.jl  a single orchestrator, no per-target branching
#
# See README.md for the invariant and the `p` (transformed) / `E` (mapped) split.

include("EigTarget.jl")
include("end_spec.jl")
include("converged_run.jl")
include("ConvergenceCriterion.jl")
include("ConvergenceState.jl")
include("calc_ritz_residuals.jl")
include("calc_eigen_residuals.jl")
include("check_convergence.jl")
