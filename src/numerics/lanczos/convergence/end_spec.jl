# ═══════════════════════════════════════════════════════════════════════════════
# THE END-SPEC INTERFACE — four methods, implemented once per `EndSpec`.
#
# This file is the whole "what does convergence mean" surface of the module. Adding a
# new spec = adding one section below. Nothing else in the module dispatches on the
# spec type: the walker (`converged_run.jl`) and the orchestrator
# (`check_convergence.jl`) are written entirely against these four methods.
#
#   1. pool_indices(spec, p, islow) -> Vector{Int}
#        Which Ritz indices are worth looking at, in WALK ORDER (extreme first).
#
#   2. needs_mapped(spec) -> Bool
#        Does the stop rule read the mapped values `E`? When it does NOT, the rule can be
#        evaluated on `p` alone and used as a free pre-gate, so eigenvectors are only
#        reconstructed once the answer could be yes. When it DOES (only `Window`), there is
#        no such shortcut and the reconstruction happens on every scheduled check.
#
#   3. is_satisfied(spec, run_p, run_E, islow) -> Bool
#        The stopping rule, evaluated on the CONTIGUOUS CONVERGED RUN.
#
#   4. select(spec, run, run_p, run_E, islow) -> Vector{Int}
#        Which members of the run the caller actually asked for. The run reaches PAST the
#        request (that overshoot is what proves completeness — the `ToValue` guard, the
#        `Window` brackets); this trims it back.
#
# `run_p` / `run_E` are the run's values in walk order (extreme first). `run_E` is
# empty when `needs_mapped(spec) == false`.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    end_walk(n::Int, islow::Bool) -> AbstractVector{Int}

Index order for walking `n` ascending Ritz values **inward from one extreme**: `1:n` at the
low end, `n:-1:1` at the high end.

Every rule in this module is expressed in this order, so *"contiguous from the extreme"*
reduces to *"a prefix"* — which is the entire reason holes become structurally impossible.
"""
end_walk(n::Int, islow::Bool) = islow ? (1:n) : (n:-1:1)

"""
    past_boundary(bound, p, islow) -> Bool

Whether the walk has stepped **past** `bound` in `p`. Walking inward, `p` increases from
the low end and decreases from the high end, so "past" flips sense with `islow`.
"""
past_boundary(bound::Real, p::Real, islow::Bool) = islow ? (p > bound) : (p < bound)

# ═══════════════════════════════════════════════════════════════════════════════
# CASE 1 — NumEigs: a fixed count.
# ═══════════════════════════════════════════════════════════════════════════════

pool_indices(s::NumEigs, p::AbstractVector, islow::Bool) =
    collect(Iterators.take(end_walk(length(p), islow), s.n))

needs_mapped(::NumEigs) = false

is_satisfied(s::NumEigs, run_p::AbstractVector, run_E::AbstractVector, islow::Bool) =
    length(run_p) >= s.n

# The pool is already capped at `n`, so the run never overshoots — nothing to trim.
select(::NumEigs, run::AbstractVector{Int}, run_p, run_E, islow::Bool) = collect(run)

# ═══════════════════════════════════════════════════════════════════════════════
# CASE 2 — ToValue: everything out to a `p` boundary, + one guard pair past it.
#
# The guard is what makes the set complete: if the run has stepped past the boundary,
# every pair inside the boundary is behind it in walk order, hence in the run.
# ═══════════════════════════════════════════════════════════════════════════════

"""
    pool_indices(s::ToValue, p, islow) -> Vector{Int}

Everything inside the boundary, plus one pair past it — the completeness guard.
"""
function pool_indices(s::ToValue, p::AbstractVector, islow::Bool)
    order = collect(end_walk(length(p), islow))
    inband = 0
    for i in order
        past_boundary(s.value, p[i], islow) && break
        inband += 1
    end
    return order[1:min(length(order), inband + 1)]
end

needs_mapped(::ToValue) = false

is_satisfied(s::ToValue, run_p::AbstractVector, run_E::AbstractVector, islow::Bool) =
    any(x -> past_boundary(s.value, x, islow), run_p)

"""
    select(s::ToValue, run, run_p, run_E, islow) -> Vector{Int}

Drop the completeness guard: the caller asked for eigenpairs out to `value`, and the pair
*past* it exists only to prove none inside were missed.
"""
select(s::ToValue, run::AbstractVector{Int}, run_p, run_E, islow::Bool) =
    Int[run[k] for k in eachindex(run) if !past_boundary(s.value, run_p[k], islow)]

# ═══════════════════════════════════════════════════════════════════════════════
# CASE 3 — Window: mapped selection, with either legacy straddle completion or a
# transformed-space completeness boundary.
#
# `Window(low, high)` is the backward-compatible mapped-straddle mode. It cannot certify
# immediate energy neighbours under an oscillatory filter tail; its brackets are explicitly
# candidates, not adjacency certificates.
#
# `Window(low, high, edge_value)` is the patch-safe mode. `edge_value` bounds a validated
# transformed passband. Converging one guard pair past it certifies the whole p-band; mapped
# values are then used only to select an overlapping energy interval.
# ═══════════════════════════════════════════════════════════════════════════════

function _take_past(p::AbstractVector, islow::Bool, bound::Real, guard::Int=1)
    order = collect(end_walk(length(p), islow))
    inband = 0
    for i in order
        past_boundary(bound, p[i], islow) && break
        inband += 1
    end
    return order[1:min(length(order), inband + guard)]
end

pool_indices(s::Window, p::AbstractVector, islow::Bool) =
    s.edge_value === nothing ? collect(end_walk(length(p), islow)) :
                               _take_past(p, islow, s.edge_value, 1)

needs_mapped(::Window) = true

"""
    is_satisfied(s::Window, run_p, run_E, islow) -> Bool

In legacy mode, the run must straddle both finite mapped bounds. Infinite bounds are
waived, but an empty run is never certified.

With `edge_value`, satisfaction instead means that the transformed run stepped one pair
past that boundary. This is the completeness guard for a validated non-leaking passband.
"""
function is_satisfied(s::Window, run_p::AbstractVector, run_E::AbstractVector, islow::Bool)
    if s.edge_value !== nothing
        return any(p -> past_boundary(s.edge_value, p, islow), run_p)
    end

    # Never certify an empty run (the live implementation did so for Window(-Inf, Inf)).
    # A nonempty all-infinite window retains the existing "both sides waived" behavior.
    isempty(run_E) && return false
    below = isinf(s.low)  || any(E -> E < s.low,  run_E)
    above = isinf(s.high) || any(E -> E > s.high, run_E)
    return below && above
end

"""
    select(s::Window, run, run_p, run_E, islow) -> Vector{Int}

The in-window pairs (bounds inclusive). When `return_brackets` is true, also retain the
nearest member *of the converged transformed run* on each finite side.

A bracket is not necessarily the immediate mapped-energy neighbour: outside a Chebyshev
passband the filter oscillates, so transformed order is not mapped-energy order. Patching
must use overlapping three-argument windows with `return_brackets=false`, not these bracket
candidates, when a strict suffix/prefix overlap is required.
"""
function select(s::Window, run::AbstractVector{Int}, run_p, run_E, islow::Bool)
    keep = Int[run[k] for k in eachindex(run) if s.low <= run_E[k] <= s.high]
    s.return_brackets || return keep

    lo_b = _nearest_bracket(run, run_E, s.low, true)
    hi_b = _nearest_bracket(run, run_E, s.high, false)
    lo_b === nothing || push!(keep, lo_b)
    hi_b === nothing || push!(keep, hi_b)
    return keep
end

"""
    _nearest_bracket(run, run_E, bound, below) -> Union{Int,Nothing}

The member of `run` strictly on one side of `bound` whose mapped value is nearest it.
`below=true` looks for `E < bound`, otherwise for `E > bound`. This is the tightest
*available* bracket candidate, not a proof of spectral adjacency.
"""
function _nearest_bracket(run::AbstractVector{Int}, run_E::AbstractVector, bound::Real, below::Bool)
    isinf(bound) && return nothing
    best, bestdist = nothing, Inf
    for k in eachindex(run)
        onside = below ? (run_E[k] < bound) : (run_E[k] > bound)
        onside || continue
        d = abs(run_E[k] - bound)
        if d < bestdist
            best, bestdist = run[k], d
        end
    end
    return best
end
