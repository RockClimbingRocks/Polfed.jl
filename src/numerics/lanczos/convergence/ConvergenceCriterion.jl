"""
    ConvergenceCriterion{T<:EigTarget, M, C<:Function}

Immutable Lanczos convergence configuration (the "what to converge and how"):

- `target::T`: which eigenpairs to target ([`EigTarget`](@ref)).
- `tol::Float64`: Ritz-residual tolerance used during iteration.
- `eigentol::Float64`: true eigen-residual tolerance (`‖A·v − λ·v‖`) for acceptance.
- `maxiter::Int`, `maxdim::Int`, `blocksize::Int`: Krylov budget.
- `mapvals::M`: callback computing the *original* operator action, used to evaluate
  eigenvalues (Rayleigh quotients) and eigen-residuals (may differ from the iteration
  operator, e.g. POLFED's filtered vs. unfiltered `H`).
- `next_check::C`: scheduling callback `(state, criterion, k) -> Int` returning the
  next iteration at which a convergence check should run. Defaults to
  [`DEFAULT_NEXT_CHECK`](@ref), which adapts the interval to the current Ritz residual.
  A custom function receives full context via `state` and `criterion` (including
  `maxdim`, `blocksize`, `maxiter`).
- `warn_incomplete::Bool`: whether to emit the WARN-level "Krylov budget exhausted before
  the target was certified complete" message on finalization. `true` (default) for callers
  that want the whole target; `false` for a caller that deliberately over-targets and keeps
  only a subset (e.g. patched POLFED converges a padded window and discards the brackets),
  where hitting `maxdim` is expected and the message is pure noise. Does **not** gate the
  numerical-breakdown (non-finite residual) warning — that always fires.

Paired with the mutable [`ConvergenceState`](@ref).
"""
struct ConvergenceCriterion{T<:EigTarget, M, C<:Function}
    target::T
    tol::Float64
    eigentol::Float64
    maxiter::Int
    maxdim::Int
    blocksize::Int
    mapvals::M
    next_check::C
    warn_incomplete::Bool
end
