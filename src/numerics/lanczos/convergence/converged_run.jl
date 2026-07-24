# ═══════════════════════════════════════════════════════════════════════════════
# THE WALKER — the one place that enforces the module's invariant:
#
#   Every selected pair comes from a CONTIGUOUS INTERNAL RUN inward from the targeted
#   transformed extreme, and EVERY member of that run passed BOTH tolerances.
#
#   No exceptions. Not on `done`, not on `maxiter`, not on breakdown.
#
# `select` may trim that run in mapped coordinates, so a legacy Window result is not
# claimed to be contiguous in mapped-energy order.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    converged_prefix(pool::AbstractVector{Int}, ok) -> Vector{Int}

The longest prefix of `pool` (already in walk order, extreme first) whose every member
satisfies `ok(i)`. The first failure ends the walk — a hole is never stepped over.

`ok` is the *combined* predicate (Ritz **and** eigen) at the point where it matters; see
[`decide_end`](@ref).
"""
function converged_prefix(pool::AbstractVector{Int}, ok)
    k = 0
    for i in pool
        ok(i) || break
        k += 1
    end
    return pool[1:k]
end

"""
    EndVerdict

Outcome of deciding one spectral end.

# Fields
- `run::Vector{Int}`: selected members of the internal contiguous converged run. Every
  returned member passed both tolerances; mapped selection may remove internal p-run members.
- `satisfied::Bool`: the spec's stopping rule is met on `run`.
- `worst_residual::Float64`: residual of the first required blocker while incomplete, or
  the worst Ritz residual over the completed internal run.
"""
struct EndVerdict
    run::Vector{Int}
    satisfied::Bool
    worst_residual::Float64
end

"""
    decide_end(spec::EndSpec, islow, p, ritz_ok, ritz, eigen_of; force=false) -> EndVerdict

Decide one spectral end. The four-stage shape below is the whole algorithm, and it is
identical for every spec — only the interface methods of `end_spec.jl` vary.

1. **Pool** — `pool_indices(spec, …)`: the indices worth looking at, in walk order.
2. **Ritz prefix** — walk from the extreme, keeping pairs while they pass the Ritz
   tolerance; stop at the first that does not. Free.
3. **Both-tolerance prefix** — reconstruct eigenpairs for that prefix, re-walk under
   `ritz_ok(i) && eigen_ok(i)` (so the run can only shrink, never gain a hole), and apply
   the stopping rule to the result.
4. **Select** — trim the run back to what was actually requested, dropping the overshoot
   that proved completeness.

Step 3 is skipped when the stopping rule reads `p` only (`!needs_mapped`) and the rule
already fails on the Ritz prefix: it cannot start passing once the run shrinks, so there is
nothing to learn and no eigenvector need be reconstructed. Window-specific mapped selection
and its optional transformed-boundary pre-gate are handled by the overload below.

# Keywords
- `force`: evaluate the run even when the pre-gate would skip it. On a forced finalization
  (`maxiter`, breakdown) there is no next check, so the partial run must be extracted now;
  short-circuiting would return an empty result instead of the pairs that did converge.

`eigen_of(idxs) -> (E, eigen_ok)` is supplied by the caller: it reconstructs the
eigenpairs for `idxs` and returns their mapped values plus a per-position acceptance
predicate. Keeping it a callback is what lets this file stay free of any factorization,
GPU or layout dependency.
"""
function decide_end(spec::EndSpec, islow::Bool, p::AbstractVector,
                    ritz_ok, ritz::AbstractVector, eigen_of; force::Bool=false)

    # 1. pool ------------------------------------------------------------------
    pool = pool_indices(spec, p, islow)
    worst = isempty(pool) ? Inf : maximum(ritz[i] for i in pool)

    # 2. Ritz prefix -----------------------------------------------------------
    cheap_run = converged_prefix(pool, ritz_ok)

    # pre-gate: only possible when the rule does not read `E`
    if !force && !needs_mapped(spec) &&
       !is_satisfied(spec, [p[i] for i in cheap_run], Float64[], islow)
        return EndVerdict(Int[], false, worst)
    end

    # 3. both-tolerance prefix -------------------------------------------------
    E, eigen_ok = eigen_of(cheap_run)
    at = Dict(cheap_run[k] => k for k in eachindex(cheap_run))
    run = converged_prefix(cheap_run, i -> ritz_ok(i) && eigen_ok(at[i]))

    run_p = [p[i] for i in run]
    run_E = needs_mapped(spec) ? [E[at[i]] for i in run] : eltype(E)[]

    satisfied = is_satisfied(spec, run_p, run_E, islow)

    # 4. select ----------------------------------------------------------------
    return EndVerdict(select(spec, run, run_p, run_E, islow), satisfied, worst)
end

"""
    decide_end(spec::Window, islow, p, ritz_ok, ritz, eigen_of; force=false) -> EndVerdict

`Window` override. In two-argument legacy mode the pool is the whole current projected
spectrum and completion is a mapped straddle. With a transformed `edge_value`, the pool ends
one pair past that boundary and the cheap p-space guard is checked before reconstruction.

`eigen_of(idxs) -> (E, eigen_ok)` is batched (one reconstruction for a whole index list, with
`eigen_ok` taking a *position within that list*), so the Ritz prefix is taken first and the
eigen check applied within it. That is the same stopping point as testing both predicates in a
single pass — the run ends at the first index failing either — but costs one reconstruction
instead of one per pair.
"""
function decide_end(spec::Window, islow::Bool, p::AbstractVector,
                    ritz_ok, ritz::AbstractVector, eigen_of; force::Bool=false)
    pool = pool_indices(spec, p, islow)
    ritz_run = converged_prefix(pool, ritz_ok)
    progress = _ritz_progress_residual(pool, ritz_run, ritz)

    # The transformed-boundary form has a genuine cheap gate. Until the Ritz prefix
    # reaches its one-pair guard, reconstructing mapped eigenpairs cannot change the answer.
    if !force && spec.edge_value !== nothing &&
       !is_satisfied(spec, view(p, ritz_run), Float64[], islow)
        return EndVerdict(Int[], false, progress)
    end

    E, eigen_ok = eigen_of(ritz_run)

    n = 0
    for k in eachindex(ritz_run)
        eigen_ok(k) || break
        n += 1
    end

    run   = ritz_run[1:n]
    run_p = [p[i] for i in run]
    run_E = E[1:n]
    satisfied = is_satisfied(spec, run_p, run_E, islow)
    residual = satisfied ? _maximum_residual(run, ritz) : progress

    return EndVerdict(select(spec, run, run_p, run_E, islow), satisfied, residual)
end

function _ritz_progress_residual(pool::AbstractVector{Int}, prefix::AbstractVector{Int},
                                 ritz::AbstractVector)
    isempty(pool) && return Inf
    if length(prefix) == length(pool)
        return maximum(ritz[i] for i in pool)
    end
    return Float64(ritz[pool[length(prefix) + 1]])
end

_maximum_residual(run::AbstractVector{Int}, ritz::AbstractVector) =
    isempty(run) ? 0.0 : maximum(ritz[i] for i in run)

"""
    Verdict

Outcome of deciding a whole [`EigTarget`](@ref) — the union over its ends.

# Fields
- `keep::Vector{Int}`: indices to return. For [`BothEnds`](@ref) this is the union of two
  contiguous runs from opposite extremes; the untouched middle of the spectrum is not a
  hole, it is simply not requested.
- `done::Bool`: every end satisfied — the iteration may stop successfully.
- `worst_residual::Float64`: worst Ritz residual across ends, for the scheduler and report.
"""
struct Verdict
    keep::Vector{Int}
    done::Bool
    worst_residual::Float64
end

"""
    decide(target::EigTarget, p, ritz_ok, ritz, eigen_of; force=false) -> Verdict

Decide a whole target by running [`decide_end`](@ref) over `ends(target)` and combining.
`Lowest`, `Highest` and `BothEnds` need no separate code paths — that is what `ends`
buys. `force` is passed through (see [`decide_end`](@ref)).
"""
function decide(target::EigTarget, p::AbstractVector,
                ritz_ok, ritz::AbstractVector, eigen_of; force::Bool=false)
    verdicts = [decide_end(spec, islow, p, ritz_ok, ritz, eigen_of; force=force)
                for (spec, islow) in ends(target)]
    return Verdict(
        unique(vcat((v.run for v in verdicts)...)),
        all(v -> v.satisfied, verdicts),
        maximum(v -> v.worst_residual, verdicts),
    )
end
