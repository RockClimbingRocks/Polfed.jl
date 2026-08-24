# resolve_filter — the interval(s) → filter orchestrator. Two mechanisms compose:
#
# 1. GEOMETRY picks the CHAIN ([`edge_side`](@ref)): a window touching a spectrum edge walks
#    `Defaults.filters.EDGE_FILTER_CHAIN`, anything interior walks `Defaults.filters.INTERIOR_FILTER_CHAIN`. A chain is an ordered
#    tuple of filter TYPES — each is tried in turn, and the first type that produces a valid
#    candidate wins; adding a new filter to the resolution = append its type to a chain.
#
# 2. Each filter type walks its OWN parameter ladder through the one generic retry loop
#    [`_first_valid`](@ref): the type declares its candidates (`_cutoff_values(F)` — cutoffs
#    0.17…0.35 for DeltaUp, the negated ladder for DeltaDown, the fixed monomial order for
#    Edge), each is built fresh (`_build_candidate`) and judged by `validate_filter` alone
#    (each filter's validation carries its full fitness contract — the deltas fold in
#    passband-covers-request + strictly-interior), first valid wins. This is what rescues
#    near-edge interior windows like [-0.95, -0.9]: the first rung's side lobe leaks past the
#    boundary and fails, a higher rung passes — the window STAYS a delta case rather than
#    escalating down the chain.
#
# A returned filter's passband may be WIDER than the requested interval (a ladder rung only has
# to COVER the request). That is fine by contract: the pipeline converges the passband but
# selects/trims on the REQUESTED interval (the Window criterion + `build_polfed_result` trim),
# so only eigenvalues inside the request are returned.

# The resolution CHAINS (which filter types are tried, in order) live in the central registry:
# `Defaults.filters.INTERIOR_FILTER_CHAIN` / `.EDGE_FILTER_CHAIN`. Appending a filter type there
# is how a new filter enters resolution.

# ── shared checks ─────────────────────────────────────────────────────────────

"""Guard a two-interval request: both well-formed and `first` strictly left of `second`."""
function _check_pair(first::Interval, second::Interval)
    _check_interval(first)
    _check_interval(second)
    first.right < second.left ||
        throw(ArgumentError("two intervals must be ordered and strictly disjoint; got " *
                            "[$(first.left), $(first.right)] and [$(second.left), $(second.right)]."))
    return nothing
end

# ── the one retry loop ────────────────────────────────────────────────────────

"""
    _first_valid(F::Type{<:AbstractFilter}, request) -> F | nothing

The one retry loop every filter type shares. Walks `F`'s own parameter ladder
([`_cutoff_values`](@ref)`(F)`): for each value, build a **fresh** candidate
([`_build_candidate`](@ref)`(F, request, value)`) and return the first that passes
[`validate_filter`](@ref) — each filter's validation carries its full fitness contract (the
deltas fold in passband-covers-request + strictly-interior; the filter stores its requested
interval, so no extra argument is needed). A candidate that throws during build/validation
(e.g. a delta passband poking past a spectrum edge) counts as invalid and the ladder moves on —
only `InterruptException` escapes. Returns `nothing` when every rung fails, which tells the
chain walk in [`resolve_filter`](@ref) to try the next filter type.

`request` is whatever `F`'s resolution methods expect: an `Interval` for the single-window
filters, the `(iv1, iv2)` tuple for [`DoubleDeltaFilter`](@ref). Every rung is traced at DEBUG:
the rejected ones from inside `validate_filter` (via [`_reject`](@ref), which knows why), the
accepted one and any build error from here.
"""
function _first_valid(F::Type{<:AbstractFilter}, request)
    for value in _cutoff_values(F)
        try
            candidate = _build_candidate(F, request, value)     # always build a fresh filter
            if validate_filter(candidate) === true
                should_log(POLFED_DEBUG_LEVEL) &&
                    polfed_log(POLFED_DEBUG_LEVEL, "filter ladder ✓ $(_filter_tag(candidate))")
                return candidate
            end
        catch error_value
            error_value isa InterruptException && rethrow()
            should_log(POLFED_DEBUG_LEVEL) &&
                polfed_log(POLFED_DEBUG_LEVEL,
                           "filter ladder ✗ $(nameof(F)) at $value — build/validation error: " *
                           sprint(showerror, error_value))
        end
    end
    return nothing
end

# ── INPUT 1: one Interval ─────────────────────────────────────────────────────

"""
    resolve_filter(interval::Interval) -> AbstractFilter

Resolve the filter for one rescaled window: [`edge_side`](@ref) picks the chain
([`Defaults.filters.EDGE_FILTER_CHAIN`](@ref) for a window touching `±1`, [`Defaults.filters.INTERIOR_FILTER_CHAIN`](@ref)
otherwise), then each filter type in the chain walks its own parameter ladder via
[`_first_valid`](@ref) — the first type that yields a valid candidate wins.

Near-edge windows like `[-0.95, -0.9]` are *deliberately* interior/delta cases — the first
ladder rungs fail their leakage validation there and a higher rung wins, never an `EdgeFilter`.
The returned filter's passband may be **wider** than `interval` (a rung only has to cover it);
downstream must select/trim on the requested interval — which the pipeline's `Window` criterion
and result trim already do.

# Throws
- `ArgumentError`: malformed interval, or the full-spectrum window (use ED).
- `ErrorException`: every filter type in the chain exhausted its ladder.
"""
function resolve_filter(interval::Interval)
    _check_interval(interval)
    side  = edge_side(interval)
    chain = side === :interior ? Defaults.filters.INTERIOR_FILTER_CHAIN : Defaults.filters.EDGE_FILTER_CHAIN
    should_log(POLFED_DEBUG_LEVEL) &&
        polfed_log(POLFED_DEBUG_LEVEL, "Resolving a filter for $(_fmt_iv(interval)).",
                   edge_side = side, chain = join(nameof.(chain), " → "))

    for F in chain
        filter = _first_valid(F, interval)
        if filter !== nothing
            should_log(POLFED_INFO_LEVEL) &&
                polfed_log(POLFED_INFO_LEVEL, "Filter resolved: $(_filter_tag(filter)).")
            return filter
        end
        polfed_log(POLFED_DEBUG_LEVEL,
                   "filter ladder: $(nameof(F)) exhausted its candidates; trying the next type in the chain.")
    end

    error("resolve_filter: no filter validated for [$(interval.left), $(interval.right)] — " *
          "the $(side === :interior ? "interior" : "edge") chain " *
          "$(join(nameof.(chain), " → ")) is exhausted.")
end

# ── INPUT 2: a vector of two Intervals ────────────────────────────────────────

"""
    resolve_filter(intervals::AbstractVector{Interval}) -> Tuple

Resolve filters for **two** disjoint windows (ordered left to right).

For two *interior* windows, first try to serve both with ONE [`DoubleDeltaFilter`](@ref)
(peaks `+1`/`−1`, one Lanczos run harvests both regions) via its own ladder
(`_cutoff_values(DoubleDeltaFilter)`, each magnitude used as `cutoff_upper = +c`,
`cutoff_lower = −c`); on success return the 1-tuple `(double_filter,)`. If every double rung
fails — or either window sits on a spectrum edge (a boundary member must get its
[`EdgeFilter`](@ref)) — resolve the two windows **independently** through the single-interval
chain and return the 2-tuple `(filter1, filter2)`.

The tuple length is the dispatch signal for the caller: one transform run vs two.

# Throws
- `ArgumentError`: not exactly two intervals, or an unordered/overlapping pair.
- `ErrorException`: propagated from the single-interval fallback when a window resolves to nothing.
"""
function resolve_filter(intervals::AbstractVector{<:Interval})
    length(intervals) == 2 ||
        throw(ArgumentError("vector input must contain exactly two intervals; got $(length(intervals))."))

    first_interval, second_interval = intervals[1], intervals[2]
    _check_pair(first_interval, second_interval)
    first_side  = edge_side(first_interval)
    second_side = edge_side(second_interval)

    if first_side === :interior && second_side === :interior
        double_filter = _first_valid(DoubleDeltaFilter, (first_interval, second_interval))
        double_filter !== nothing && return (double_filter,)
        polfed_log(POLFED_DEBUG_LEVEL,
                   "filter ladder: no DoubleDeltaFilter validated; resolving the two windows separately.")
    else
        polfed_log(POLFED_DEBUG_LEVEL,
                   "filter ladder: a spectrum edge is present; resolving the two windows separately.")
    end

    return (resolve_filter(first_interval), resolve_filter(second_interval))
end
