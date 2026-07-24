# Transform slice: the spectral transformation — the POLFED bottleneck. Builds the
# transformed operator `f!_transformed` that Lanczos runs on, so the target eigenvalues
# become extremal. Every Lanczos matvec applies the order-`K` filter = `K` sparse matvecs
# of the rescaled `H̃` (Clenshaw recurrence).
#
# All resolution has already happened upstream: the target became an `Interval` (target/),
# and the filter was picked, built, order-resolved, and validated (filter/). So the filter
# arriving here is fully resolved. What remains — and all this slice does — is turn that
# filter + the mapping into the operator: build the Clenshaw expansion (clenshaw/) and apply
# it across the block under the mapping's parallelization strategy (parallel/).
#
# Include order is dependency-first: build the expansion, then the workspace + parallel
# apply it feeds, then the orchestrator that wires them into `f!_transformed`.

include("clenshaw/build_expansion.jl")   # _build_expansion + normalize_clenshaw_recurrence
include("parallel/b_storage.jl")         # get_b_storage (+ shared buffer, per-column triplets)
include("parallel/clenshaw_apply.jl")    # clenshaw_apply — strategy-dispatched block apply

"""
    build_transform(filter::AbstractFilter, mp::MappingPlan) -> Function

Build the transformed in-place operator `f!_transformed(Y, X)` that Lanczos runs on: the
resolved filter polynomial of the rescaled Hamiltonian `H̃`, evaluated by Clenshaw
recurrence and applied under the mapping's parallelization strategy.

Steps:
1. take the filter's **normalized** coefficients ([`coefficients`](@ref)) and order
   ([`polynomial_order`](@ref)) — the filter must be resolved (done by [`resolve_filter`](@ref)),
2. build the operator [`ClenshawExpansion`](@ref) over `mp.f!_rescaled` (or the mapping's
   custom Clenshaw kernels, if it carries any) via [`_build_expansion`](@ref),
3. allocate the recurrence workspace ([`get_b_storage`](@ref)) for the strategy, and
4. return a closure applying the expansion under that strategy ([`clenshaw_apply`](@ref)).

# Arguments
- `filter::AbstractFilter`: the resolved filter (target and polynomial order set) from
  [`resolve_filter`](@ref).
- `mp::MappingPlan`: the resolved mapping — provides `f!_rescaled`, the optional custom
  Clenshaw kernels, the parallelization strategy, and (via `mp.sys`) the starting block
  `x0`, processing unit, dimension, and element type. No separate `x0`/`pu` argument is
  needed: they live on the plan.

# Returns
The in-place transformed operator `f!_transformed(Y, X)` (returns `nothing`), ready to hand
to Lanczos. The already-resolved `target`/`order`/`left`/`right`/`expose` are *not* bundled
here — they live on the `filter` and the `Interval` the caller already holds.
"""
function build_transform(filter::AbstractFilter, mp::MappingPlan)
    x0     = mp.sys.x0
    pu     = mp.sys.pu
    order  = polynomial_order(filter)::Int
    order > 0 || throw(ArgumentError(
        "build_transform requires a resolved filter with positive polynomial order; " *
        "got order=$order for $(typeof(filter)).",
    ))
    family = polynomial_type(filter)
    dim    = size(x0, 1)
    E      = eltype(x0)

    # The order IS the per-matvec cost multiplier (one H·v per Chebyshev term), so it is the
    # single most useful number to see when a run is unexpectedly slow.
    polfed_log(POLFED_INFO_LEVEL, "Building the spectral transformation (Clenshaw filter of H̃).",
               filter = nameof(typeof(filter)), order = order, family = family,
               dim = dim, eltype = E, parallel = typeof(mp.parallel_strategy))

    coeffs_func = coefficients(filter)                       # normalized: peak at ±1 at the target
    expansion = _build_expansion(family, order, coeffs_func, mp, dim, E, x0)
    b_storage = get_b_storage(mp.parallel_strategy, pu, x0)

    polfed_log(POLFED_DEBUG_LEVEL, "Spectral transformation built.",
               order = order, b_storage = typeof(b_storage))

    return (Y::AbstractVecOrMat{<:Number}, X::AbstractVecOrMat{<:Number}) -> begin
        clenshaw_apply(expansion, Y, X, b_storage, pu, mp.parallel_strategy)
        return nothing
    end
end
