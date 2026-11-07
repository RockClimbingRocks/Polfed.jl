# Two-window strategy resolution: given two disjoint requested windows, decide whether ONE joint
# `DoubleDeltaFilter` run or TWO independent single-filter runs is cheaper, and return the filters
# for whichever path won.
#
# This is the one place the cost model lives. It sits in the filter slice (not the pipeline)
# because everything it needs is filter geometry — the resolved polynomial orders — and because
# resolving those orders is FREE relative to the run: order resolution is scalar Chebyshev
# bisection, no matvecs. So the decision is made on real numbers, never on estimates.
#
# The decision is ADVISORY, the validation is authoritative: if the double filter cannot be built
# (its ladder fails, or a window touches a spectrum edge where an EdgeFilter is mandatory) the
# strategy falls back to two runs, whatever the ratio said. `resolve_filter(::AbstractVector)`
# implements the same fallback without the cost model; this function is the cost-aware version and
# additionally reports WHY it chose what it chose.

"""
    DoubleStrategy

The resolved plan for a two-window POLFED run: which path to take and the filters it needs.

# Fields
- `mode::Symbol`: `:double` (one joint [`DoubleDeltaFilter`](@ref) run) or `:separate` (two runs).
- `double_filter`: the resolved joint filter, or `nothing` on the `:separate` path.
- `filter1`, `filter2`: the single-window filters for the LOWER and HIGHER window. Always
  resolved (their orders are what the decision is made on), and used as-is when `mode` is
  `:separate`.
- `order1::Int`, `order2::Int`: the single-window polynomial orders `K₁`, `K₂`.
- `ratio::Float64`: `min(K₁,K₂) / max(K₁,K₂)` — the decision variable.
- `reason::Symbol`: why this mode was chosen — `:ratio` (the cost model), `:forced` (the user's
  `strategy` keyword), `:edge` (a window touches `±1`, so a joint filter is impossible), or
  `:double_failed` (the joint filter's ladder found no valid candidate).
"""
struct DoubleStrategy{D<:Union{Nothing,DoubleDeltaFilter}, F1<:AbstractFilter, F2<:AbstractFilter}
    mode::Symbol
    double_filter::D
    filter1::F1
    filter2::F2
    order1::Int
    order2::Int
    ratio::Float64
    reason::Symbol
end

"""
    double_order(strategy::DoubleStrategy) -> Int

The polynomial order actually paid per Lanczos matvec on the joint path (`K_d`), or `0` on the
`:separate` path (where the two runs pay `order1` and `order2` separately).
"""
double_order(strategy::DoubleStrategy) =
    strategy.double_filter === nothing ? 0 : polynomial_order(strategy.double_filter)

"""
    resolve_double_strategy(interval1, interval2, strategy=:auto) -> DoubleStrategy

Choose between one joint double-delta run and two single-filter runs for the two rescaled windows
`interval1` (lower) and `interval2` (higher), and resolve the filters for the chosen path.

# The decision (`strategy = :auto`)
Both windows are first resolved through the ordinary single-window chain
([`resolve_filter`](@ref)), giving the orders `K₁`, `K₂` — this costs no matvecs, and the filters
are reused if the `:separate` path wins. With `r = min(K₁,K₂)/max(K₁,K₂)`:

- `r ≥ `[`Defaults.double_polfed.double_ratio`](@ref) → try the joint filter. Both windows then
  pay `K_d = max(K₁,K₂)` per matvec, but in ONE Krylov space.
- `r < double_ratio` → two runs: the cheap window's `K` is so much smaller than the expensive
  one's that making it pay `K_d` costs more than a second factorization.

See [`Defaults.double_polfed.double_ratio`](@ref) for the cost model this threshold comes from.

# Overrides and fallbacks
`strategy` may be `:auto`, `:double` (force the joint run) or `:separate` (force two runs). The
joint path is only *attempted* — never assumed: if either window touches a spectrum edge (`±1`,
where a monotone [`EdgeFilter`](@ref) is mandatory) or the double filter's cutoff ladder yields no
valid candidate, the result is `:separate` with the corresponding `reason`. Forcing `:double` and
having it fall back emits a warning; the automatic fallbacks log at INFO.

# Arguments
- `interval1::Interval`, `interval2::Interval`: the two rescaled windows, **ordered** —
  `interval1.right < interval2.left` (guarded by `_check_pair`).
- `strategy::Symbol`: `:auto`, `:double`, or `:separate`.

# Keywords

# Returns
- [`DoubleStrategy`](@ref) carrying the mode, the filters for that mode, and the decision numbers.

# Throws
- `ArgumentError`: unknown `strategy`, or windows that are malformed/unordered/overlapping.
- `ErrorException`: propagated from [`resolve_filter`](@ref) when a window resolves to no filter.
"""
function resolve_double_strategy(interval1::Interval, interval2::Interval,
                                 strategy::Symbol = Defaults.double_polfed.strategy)
    strategy in (:auto, :double, :separate) || throw(ArgumentError(
        "double_polfed: unknown strategy :$strategy (expected :auto, :double, or :separate)."))
    _check_pair(interval1, interval2)

    # Single-window filters: the decision variable, and the `:separate` path's filters. Free
    # relative to the run — order resolution is scalar bisection, not matvecs.
    filter1 = resolve_filter(interval1)
    filter2 = resolve_filter(interval2)
    K1, K2 = polynomial_order(filter1), polynomial_order(filter2)
    ratio  = min(K1, K2) / max(K1, K2)

    separate(reason::Symbol) =
        DoubleStrategy(:separate, nothing, filter1, filter2, K1, K2, ratio, reason)

    strategy === :separate && return separate(:forced)

    interior = edge_side(interval1) === :interior && edge_side(interval2) === :interior
    wants_double = strategy === :double || ratio >= Defaults.double_polfed.double_ratio

    if !wants_double
        polfed_log(POLFED_INFO_LEVEL,
                   "double POLFED: two separate runs are cheaper than one joint double-delta run.",
                   K1 = K1, K2 = K2, ratio = ratio,
                   threshold = Defaults.double_polfed.double_ratio)
        return separate(:ratio)
    end

    if !interior
        polfed_log(strategy === :double ? POLFED_WARN_LEVEL : POLFED_INFO_LEVEL,
                   "double POLFED: a requested window touches a spectrum edge (±1), where a " *
                   "monotone EdgeFilter is required — resolving the two windows separately.",
                   side1 = edge_side(interval1), side2 = edge_side(interval2))
        return separate(:edge)
    end

    joint = _first_valid(DoubleDeltaFilter, (interval1, interval2))
    if joint === nothing
        polfed_log(strategy === :double ? POLFED_WARN_LEVEL : POLFED_INFO_LEVEL,
                   "double POLFED: no DoubleDeltaFilter passed validation for this window pair " *
                   "(its cutoff ladder is exhausted) — resolving the two windows separately.",
                   interval1 = (interval1.left, interval1.right),
                   interval2 = (interval2.left, interval2.right))
        return separate(:double_failed)
    end

    polfed_log(POLFED_INFO_LEVEL,
               "double POLFED: one joint double-delta run serves both windows.",
               K1 = K1, K2 = K2, ratio = ratio, K_double = polynomial_order(joint),
               threshold = Defaults.double_polfed.double_ratio)
    return DoubleStrategy(:double, joint, filter1, filter2, K1, K2, ratio,
                          strategy === :double ? :forced : :ratio)
end
