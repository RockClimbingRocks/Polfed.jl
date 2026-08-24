# ── filter/ — per-filter layout ───────────────────────────────────────────────────────────────
#
# The slice has three parts, and the folder names say which is which:
#
#   filter.jl        this file — the `AbstractFilter` seam
#   resolution/      how a filter gets PICKED (chain walk, order bisection, window geometry)
#   filter_types/    the concrete filters that exist, one self-contained folder each
#
# Filters are organized PER FILTER, not per concern: everything about one filter (its type,
# coefficients, order/build, validation) lives in its own folder under `filter_types/`, so adding
# a filter = drop a folder and implement the interface — no editing of shared concern-folders.
#
# This file is the shared top: the `AbstractFilter` seam (interface function stubs every filter
# adds methods to), the one genuinely-shared evaluation primitive `_scalar_filter`, and the
# includes. Everything that reads a filter's own parameters lives with that filter.
#
# Shared here / cross-filter:            Per filter (filter_types/<name>/):
#   AbstractFilter + interface stubs       the struct + type-tag methods
#   _scalar_filter (evaluate P(x))         coefficients / _coefficients_nonnormalized
#   bisect_order (order_bisection.jl)      exposed_end / expected_krylov_dim / polynomial_type
#   edge_side (window geometry)            analytical_order / build_filter / resolve_order!
#   resolve_filter (cutoff ladder)         resolve_bounds_from_order / validate_filter
#
# NOTE vs the current layout: `peak_value` and `_window_bounds` are gone — each filter knows its
# own peak sign (±1) and, being symmetric, its own window (`target ± δ`); those abstractions only
# earned their keep for edge/asymmetric filters that don't exist yet (reintroduce them then).

"""
    AbstractFilter

Supertype for POLFED spectral filters. A concrete filter is a self-contained folder under
`filter_types/` implementing this interface; the resolution/transform machinery only ever calls
through it.

Interface a filter implements (all dispatched on the filter type):
- [`coefficients`](@ref)`(filter)` — normalized coefficient function `n -> cₙ` (peak at ±1).
- [`polynomial_type`](@ref)`(filter)` — polynomial family (`:Chebyshev`).
- [`polynomial_order`](@ref)`(filter)` — highest degree used by the filter polynomial.
- [`exposed_end`](@ref)`(filter)` — which spectral extreme (`:highest`/`:lowest`) the target maps to.
- [`expected_krylov_dim`](@ref)`(filter, howmany, blocksize, η)` — predicted Krylov dimension.
- [`analytical_order`](@ref)`(filter, δ)` — heuristic order bracket for a half-width `δ`.
- [`build_filter`](@ref)`(::Type{F}, interval)` — construct + resolve the filter for a window.
- [`resolve_order!`](@ref)`(filter)` / [`resolve_bounds_from_order`](@ref)`(filter)` — order ⇄ passband.
- [`validate_filter`](@ref)`(filter)` — leakage check toward the exposed end.

See `filter_types/delta_up/` for the reference implementation.
"""
abstract type AbstractFilter end

# Interface stubs — declared here so each filter's file can add methods to the SAME generic
# functions without an `import` (all filters are plain includes in `module UniPolfed`).
function coefficients end
function polynomial_type end

"""
    polynomial_order(filter::AbstractFilter) -> Int

Return the highest polynomial degree required to evaluate `filter`. A filter
with one expansion returns its resolved `order`; a composite filter returns the
highest degree present in its combined coefficient sequence.
"""
function polynomial_order end
function exposed_end end
function expected_krylov_dim end
function analytical_order end
function build_filter end
function resolve_order! end
function resolve_bounds_from_order! end
function validate_filter end

"""
    order_safety_factor(filter::AbstractFilter) -> Float64

The safety factor that was applied when the filter's polynomial order was *resolved* (the delta
filters shrink their bisected order by it to widen the passband). Filters with a fixed,
user-chosen order (e.g. `EdgeFilter`) resolve nothing, so the generic fallback reports `1.0`.
Read by the report — filters that resolve an order add a method returning their own field.
"""
order_safety_factor(::AbstractFilter) = 1.0

"""
    passband(filter::AbstractFilter) -> Interval

The filter's **passband** in rescaled coordinates: the region where the filter is trustworthy — a
delta filter's resolved `[left, right]` (where `|P| ≥ cutoff`, monotone in `|E − centre|`), or a
monotone [`EdgeFilter`](@ref)'s exposed spectrum side (unbounded toward its edge). An eigenvalue
OUTSIDE the passband sits in the filter's oscillating tail and is **not** a certified result —
callers use this as a safety gate to discard such pairs (e.g. a `Window` target's brackets, which
lie outside the passband by construction). The generic fallback `(-Inf, Inf)` is permissive (no
gating) for filters that do not resolve a passband; delta and edge filters override it.
"""
passband(::AbstractFilter) = Interval(-Inf, Inf)

"""
    cutoff(filter::AbstractFilter) -> Float64

The filter's resolved suppression **cutoff** — the `|P|` threshold whose crossings define the
passband edges (a delta filter's resolved `cutoff` field, e.g. `0.17`). Filters with no cutoff
(a monotone [`EdgeFilter`](@ref) suppresses nothing) return `NaN`. Read by the report.
"""
cutoff(::AbstractFilter) = NaN

"""
    filter_target(filter::AbstractFilter, interval::Interval) -> Float64

The rescaled center the filter's peak sits at, for the window `interval` — what the report prints
as the target energy. Single-peak filters carry it in their `target` field and ignore `interval`
(the generic method below); a composite filter with more than one peak (e.g.
[`DoubleDeltaFilter`](@ref)) uses `interval` to say WHICH peak is being reported, since one report
section is built per requested window.
"""
filter_target(filter::AbstractFilter, ::Interval) = Float64(filter.target)

# ── resolution interface (consumed by resolve_filter's chain walk) ────────────
# Each filter TYPE declares its own parameter candidates and how to consume one; the shared
# `_first_valid` loop in resolve_filter.jl then works for every filter without knowing any of
# them. Fitness is judged by `validate_filter(candidate)` alone — a filter that needs stricter
# conditions (the deltas: passband covers the requested `filter.interval` + strictly interior)
# folds them INTO its `validate_filter`; the filter carries its request, so no extra argument is
# needed. Adding a filter to a resolution chain = implement these two in the filter's own folder.

"""
    _cutoff_values(::Type{F}) -> iterable

The filter type's OWN parameter candidates, tried in order by [`_first_valid`](@ref) — **first
valid wins**, so sharpest/preferred first. What a "value" means is the filter's business: the
delta family ladders its `cutoff` (DeltaUp `0.17…0.35`, DeltaDown the negated ladder), the edge
filter its fixed monomial `order`. Declared per filter next to its `build_filter`.
"""
function _cutoff_values end

"""
    _build_candidate(::Type{F}, request, value) -> F

Build one fresh candidate of type `F` for `request` (an `Interval`, or the two-interval `Tuple`
for a double filter) at the ladder parameter `value` — the thin adapter that turns the generic
ladder value into that filter's own `build_filter` keyword(s).
"""
function _build_candidate end

"""
    _scalar_filter(filter::AbstractFilter) -> ClenshawExpansion

The filter's normalized scalar polynomial `P(x)` (peaks at ±1 at the filter's target), built
purely from the interface (`polynomial_type`, `coefficients`, and `polynomial_order`).
Callable as `P(x)`; used by every filter's order/passband resolution and leakage check. This is
the one evaluation primitive worth sharing — otherwise every filter would rewrite it.
"""
_scalar_filter(filter::AbstractFilter) =
    ClenshawExpansion(
        polynomial_type(filter),
        coefficients(filter),
        polynomial_order(filter)::Int,
    )

# ── ladder diagnostics (DEBUG tracing of the filter resolution) ───────────────
# One line per candidate, emitted where the verdict is actually known: `_reject` from inside the
# filter's own `validate_filter` (which is the only place that knows WHY a rung failed), the
# accept/error lines from `_first_valid`. Both go through `_filter_tag`, so every rung — rejected,
# accepted, or from any filter type — prints in the same shape.

"""
    _filter_tag(filter::AbstractFilter) -> String

One-line identity of a filter candidate for the ladder trace: type, resolved order, cutoff (only
if it has one — an [`EdgeFilter`](@ref) does not), and passband (only if bounded). Built lazily —
callers gate it on [`should_log`](@ref).
"""
function _filter_tag(filter::AbstractFilter)
    tag = string(nameof(typeof(filter)), " K=", polynomial_order(filter))
    c = cutoff(filter)
    isnan(c) || (tag *= @sprintf(" cutoff=%.4g", c))
    band = passband(filter)
    return isfinite(band.left) && isfinite(band.right) ? tag * " passband=" * _fmt_iv(band) : tag
end

"""
    _reject(filter::AbstractFilter, why::AbstractString) -> false

Log one rejected ladder rung at DEBUG (`filter tag — why`) and return `false`. Every
[`validate_filter`](@ref) failure path returns through this instead of a bare `false`, so the
resolution trace says which rung failed *and* what disqualified it.
"""
function _reject(filter::AbstractFilter, why::AbstractString)
    should_log(POLFED_DEBUG_LEVEL) &&
        polfed_log(POLFED_DEBUG_LEVEL, "filter ladder ✗ $(_filter_tag(filter)) — $why")
    return false
end

"""
    resolve_eig_target(exposed::Symbol, howmany::Integer) -> EigTarget

Translate a filter's [`exposed_end`](@ref) into the Lanczos [`EigTarget`](@ref) the transformed
problem should select:

- `:highest` → [`HighestEigs`](@ref)`(howmany)`
- `:lowest`  → [`LowestEigs`](@ref)`(howmany)`
- `:both`    → [`BothEnds`](@ref)`(howmany, howmany)` — `howmany` from **each** extreme (both
  windows of a double filter).

Throws on an unknown symbol.
"""
function resolve_eig_target(exposed::Symbol, howmany::Integer)
    exposed === :highest && return HighestEigs(Int(howmany))
    exposed === :lowest  && return LowestEigs(Int(howmany))
    exposed === :both    && return BothEnds(Int(howmany), Int(howmany))
    throw(ArgumentError("unknown exposed_end :$exposed (expected :highest, :lowest, or :both)"))
end





"""
    window_eig_target(filter::AbstractFilter, interval::Interval) -> EigTarget

The Lanczos [`EigTarget`](@ref) for the **window** pipeline: a [`Window`](@ref) over the target
interval, so it converges *exactly* the eigenpairs inside `[lo, hi]` plus a bracket just past each
edge — not the `osf`-widened passband a cutoff threshold would catch.

**Units.** The window is stated in **rescaled** coordinates, the same space `interval` and the
filter live in, so callers must pass `mapvals = mp.f!_rescaled` — a `Window` is always tested
against whatever `mapvals` produces. Nothing is mapped back to true energy here; the pipeline
un-rescales once, at the result boundary.

- `:highest` → `Highest(Window(lo, hi))` — the interval maps to the transformed high end.
- `:lowest`  → `Lowest(Window(lo, hi))`  — the interval maps to the transformed low end.

The criterion needs nothing from the filter beyond which end it exposes: its rule lives entirely
in mapped-energy space (see [`Window`](@ref)), with no `p`-space threshold to derive.

A bound sitting on a rescaled spectrum edge (`±1`, e.g. an edge-filter window) means the window
runs past the end of the spectrum: no eigenvalue exists beyond it, so no bracket can — or need —
converge there. That side is encoded as `±Inf`, which the `Window` criterion treats as
bracket-free (see [`Window`](@ref)); without this the stop test would wait forever for a
nonexistent bracket and always exhaust `maxdim`.

Contrast [`resolve_eig_target`](@ref), the count-based ([`NumEigs`](@ref)) target for the
point/count pipeline. Called once at setup, so the symbol-dispatched return type is fine.

# Throws
- `ArgumentError` on `:both` (double-filter windows — not yet supported) or an unknown
  [`exposed_end`](@ref).
"""
function window_eig_target(filter::AbstractFilter, interval::Interval)
    lo = interval.left  <= -1.0 + 1e-9 ? -Inf : interval.left
    hi = interval.right >=  1.0 - 1e-9 ?  Inf : interval.right

    e = exposed_end(filter)
    e === :highest && return Highest(Window(lo, hi))
    e === :lowest  && return Lowest(Window(lo, hi))
    e === :both    && throw(ArgumentError("window_eig_target: :both (double-filter windows) not yet supported."))
    throw(ArgumentError("window_eig_target: unknown exposed_end :$e (expected :highest or :lowest)"))
end











include("resolution/order_bisection.jl")  # shared: bisect_order (bracket-double + bisect, predicate-driven)
include("resolution/edge_side.jl")        # shared: window geometry (edge_side — the family decision)
include("filter_types/filter_types.jl")   # the concrete filters (one folder each)
include("resolution/resolve_filter.jl")   # shared: geometry + cutoff-ladder orchestrator (single & pair)
include("resolution/resolve_double_strategy.jl")  # shared: cost-aware joint-vs-separate decision (two windows)
