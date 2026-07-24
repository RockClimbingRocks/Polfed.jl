# EdgeFilter — the monomial power filter for windows at the spectrum EDGES. Where the delta
# filters concentrate an interior window with a high-order Chebyshev peak, the edge filter is a
# low-order power of an affine map of the rescaled H̃ ("basically H³"):
#
#   :lowest   P(x) = −((1−x)/2)ᵖ    left-edge window  → transformed lowest  (P(−1) = −1, P(+1) = 0)
#   :highest  P(x) = +((1+x)/2)ᵖ    right-edge window → transformed highest (P(−1) = 0,  P(+1) = +1)
#   :both     P(x) = xᵖ, p odd      both edges at once → both transformed extremes (P(±1) = ±1)
#
# All three are MONOTONE INCREASING on [-1, 1]. That is the whole design: the target edge is
# stretched (slope ~p/2 at the exposed end) while the rest of the spectrum is squashed toward 0,
# ordering is preserved (so the Window stop criterion and the mapped-value selection work
# unchanged), and — being monotone — the polynomial HAS no side lobes, so the boundary leakage
# that disqualifies the delta filters near ±1 (see `RESTRUCTURE.md` §7) cannot occur.
#
# Consequently almost none of the delta machinery is needed: no order bisection (the order is a
# fixed cost/stretch knob, default 3), no passband growth (a monotone filter's effective window IS
# the requested window), and validation reduces to a cheap monotonicity/endpoint sanity scan.
#
# This file: the filter's own defaults, its (immutable) struct + constructor, and the trivial
# type-tag methods. The behavior is split into sibling files, included at the bottom:
#   coefficients.jl — the Taylor (monomial) coefficient table (exact ±1 peaks, no normalization)
#   build.jl        — build_filter from window geometry (single edge and the two-interval :both)
#   validate.jl     — the monotonicity/endpoint sanity scan (a monotone filter cannot leak)

"""Per-filter defaults for [`EdgeFilter`](@ref) (namespaced like every `<Name>FilterDefaults`)."""
module EdgeFilterDefaults
    "Default monomial power `p`. Each Lanczos matvec costs `p` matvecs of `H̃`; higher `p`
    stretches the edge more (fewer iterations) at more matvecs per iteration."
    const order = 3

    "Grid size of the `validate_filter` monotonicity/endpoint sanity scan."
    const npts_validate = 513

    # Krylov-dimension estimate `ceil((iters_per_block·bs + iters_per_eig·howmany)·η)`.
    # INITIAL estimates, not an empirical fit (unlike the delta constants 20.427/1.696): the edge
    # filter's polynomial amplification is weaker than a delta peak, so `iters_per_eig` is set
    # higher. Refine from real runs when data accumulates.
    const krylov_iters_per_block = 25.0
    const krylov_iters_per_eig   = 4.0
    "Extra budget factor when `:both` ends converge in one Krylov space (cf. DoubleDelta's 1.30)."
    const krylov_both_factor     = 1.30
end

"""
    EdgeFilter(interval, exposed_end; order) <: AbstractFilter

Monomial power filter for spectrum-EDGE windows: a monotone increasing polynomial that stretches
the exposed edge(s) of the rescaled spectrum and squashes the rest toward `0`, at a cost of only
`order` matvecs per Lanczos iteration. The shape follows `exposed_end`:

- `:lowest` — `P(x) = −((1−x)/2)^order`: a left-edge window maps to the transformed **lowest** end.
- `:highest` — `P(x) = ((1+x)/2)^order`: a right-edge window maps to the transformed **highest** end.
- `:both` — `P(x) = x^order` (`order` **odd**): both spectrum edges map to both transformed
  extremes at once (a [`BothEnds`](@ref) Lanczos run harvests them in a single Krylov space).

Because `P` is monotone it preserves the spectral ordering (mapped-value selection and the
`Window` stop criterion work unchanged) and cannot leak — there are no side lobes — so
[`validate_filter`](@ref) is a plain sanity scan and there is no order/passband resolution.
The struct is immutable: everything is fixed at construction.

Prefer [`build_filter`](@ref)`(EdgeFilter, interval)` (derives `exposed_end` from which spectrum
edge the window touches) or `build_filter(EdgeFilter, interval1, interval2)` (the `:both` form);
this constructor is the explicit-`exposed_end` core they share.

# Fields
- `interval::Interval`: the requested rescaled window (for `:both`: the hull of the two edge
  windows). Not consumed by the pipeline — kept for validation and diagnostics.
- `target::Float64`: the exposed rescaled edge, `-1.0`/`+1.0` — or `NaN` for `:both`, which has
  no single target (only reachable through the report if a future multi-window pipeline path
  renders it).
- `order::Int`: the monomial power `p ≥ 1` (odd for `:both`). This IS the polynomial order — a
  fixed knob, never resolved.
- `exposed_end::Symbol`: `:lowest` / `:highest` / `:both`.

# Throws
- `ArgumentError`: `order < 1`, an unknown `exposed_end`, or an even `order` with `:both`
  (an even power folds both edges onto the same end and breaks monotonicity).
"""
struct EdgeFilter <: AbstractFilter
    interval::Interval
    target::Float64
    order::Int
    exposed_end::Symbol

    function EdgeFilter(interval::Interval, exposed_end::Symbol;
                        order::Integer = EdgeFilterDefaults.order)
        order >= 1 ||
            throw(ArgumentError("EdgeFilter: order must be >= 1, got $order."))
        exposed_end in (:lowest, :highest, :both) ||
            throw(ArgumentError("EdgeFilter: exposed_end must be :lowest, :highest, or :both; got :$exposed_end."))
        exposed_end === :both && iseven(order) &&
            throw(ArgumentError("EdgeFilter: :both requires an ODD order (P(x) = xᵖ must stay " *
                                "monotone with P(±1) = ±1); got order = $order."))
        target = exposed_end === :lowest ? -1.0 : exposed_end === :highest ? 1.0 : NaN
        return new(interval, target, Int(order), exposed_end)
    end
end

# ── interface: type tags ──────────────────────────────────────────────────────

"""    exposed_end(filter::EdgeFilter) -> Symbol — `:lowest`, `:highest`, or `:both` (stored)."""
exposed_end(filter::EdgeFilter) = filter.exposed_end

"""    passband(f::EdgeFilter) -> Interval — the exposed spectrum side, unbounded toward its edge.

Monotone (no cutoff), so the passband is the whole exposed half: `(-Inf, 1]` for `:lowest`,
`[-1, Inf)` for `:highest`, `(-Inf, Inf)` for `:both`. The unbounded side lets the extreme
eigenvalue (which can map just past `±1`, since `Emin`/`Emax` are estimates) survive the gate,
while the finite side still catches a grossly out-of-range pair."""
passband(f::EdgeFilter) = f.exposed_end === :lowest  ? Interval(-Inf, 1.0) :
                          f.exposed_end === :highest ? Interval(-1.0, Inf) :
                                                       Interval(-Inf, Inf)

"""    polynomial_type(::EdgeFilter) -> :Taylor — monomial basis; Clenshaw degenerates to Horner."""
polynomial_type(::EdgeFilter) = :Taylor

"""    polynomial_order(filter::EdgeFilter) -> Int — the monomial power `p` (fixed at construction)."""
polynomial_order(filter::EdgeFilter) = filter.order

"""
    expected_krylov_dim(filter::EdgeFilter, howmany, blocksize, η) -> Int

Predicted Krylov dimension `ceil((25·blocksize + 4·howmany)·η)` (constants from
[`EdgeFilterDefaults`](@ref)), scaled by `1.30` for `:both` (both extremes share one Krylov
space, cf. the DoubleDelta budget). These are **initial estimates**, not an empirical fit like
the delta filter's `20.427/1.696` — the edge filter's polynomial amplification is weaker than a
delta peak, so the per-eigenpair cost is budgeted higher until real-run data pins it down.
"""
expected_krylov_dim(filter::EdgeFilter, howmany::Int, blocksize::Int, η::Real) =
    ceil(Int, (EdgeFilterDefaults.krylov_iters_per_block * blocksize +
               EdgeFilterDefaults.krylov_iters_per_eig * howmany) *
              (filter.exposed_end === :both ? EdgeFilterDefaults.krylov_both_factor : 1.0) * η)

include("coefficients.jl")
include("build.jl")
include("validate.jl")
