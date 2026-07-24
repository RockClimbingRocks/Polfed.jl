# Build the operator ClenshawExpansion for the transform: the filter polynomial of the
# rescaled Hamiltonian, evaluated by Clenshaw recurrence over `mp.f!_rescaled`. This file
# is purely about *which* polynomial and *which* Clenshaw kernels — the block-splitting
# (workspace + parallel apply) lives in transform/parallel/. All resolution (target,
# bounds, order, validation) already happened upstream in target/ and filter/, so the
# filter arriving here is fully resolved (`target` and `order` set).

"""
    _build_expansion(family, order, coefficients, mp, dim, E, x0) -> ClenshawExpansion

Build the operator [`ClenshawExpansion`](@ref) for the transform, selecting the Clenshaw
kernels from the [`MappingPlan`](@ref):

- neither `mp.clenshaw_recurrence` nor `mp.clenshaw_finalsum` set (the usual case) → derive
  both from `mp.f!_rescaled`,
- both set → use them directly,
- only one set → use it and derive the other from `mp.f!_rescaled` via
  [`build_clenshaw_kernels`](@ref).

A user recurrence is adapted to the internal six-argument form by
[`normalize_clenshaw_recurrence`](@ref).

# Arguments
- `family::Symbol`: polynomial family (e.g. `:Chebyshev`) taken from the filter.
- `order::Int`: resolved polynomial order (from [`polynomial_order`](@ref)).
- `coefficients`: the filter's **normalized** coefficient function `n -> cₙ` (from
  [`coefficients`](@ref)).
- `mp::MappingPlan`: carries `f!_rescaled` and the optional custom Clenshaw kernels.
- `dim::Int`, `E::Type`: operator dimension and element type (from `x0`).
- `x0::AbstractVecOrMat`: probe used to detect a user recurrence's arity.

# Returns
A [`ClenshawExpansion`](@ref) callable as `expansion(Y, X, b)`.
"""
function _build_expansion(family::Symbol, order::Int, coefficients,
                          mp::MappingPlan, dim::Int, ::Type{E}, x0::AbstractVecOrMat) where {E}
    has_recurrence = !isnothing(mp.clenshaw_recurrence)
    has_finalsum   = !isnothing(mp.clenshaw_finalsum)

    if !has_recurrence && !has_finalsum
        return ClenshawExpansion(family, coefficients, order, mp.f!_rescaled, dim, E)
    end

    recurrence = if has_recurrence
        normalize_clenshaw_recurrence(mp.clenshaw_recurrence, x0, mp.parallel_strategy)
    else
        first(build_clenshaw_kernels(mp.f!_rescaled, family))
    end
    finalsum = has_finalsum ? mp.clenshaw_finalsum : last(build_clenshaw_kernels(mp.f!_rescaled, family))

    return ClenshawExpansion(family, coefficients, order, recurrence, finalsum, dim, E)
end

"""
    normalize_clenshaw_recurrence(recurrence, x0, strategy::Parallelization) -> Function

Adapt a user Clenshaw recurrence to the internal six-argument signature
`(b1, b2, b3, c, k, X)`. Accepts either that form or the five-argument
`(b1, b2, b3, c, X)` (the step index `k` is then ignored). The `strategy` selects the
sample used to probe which form the callback accepts (a single column view for the
block-parallel strategies, the whole `x0` for [`NoParallel`](@ref)).

# Throws
- `ArgumentError`: if `recurrence` accepts neither signature.
"""
function normalize_clenshaw_recurrence(recurrence, x0::AbstractVecOrMat, strategy::Parallelization)
    sample = if strategy isa NoParallel
        x0
    else
        x0 isa AbstractVector ? x0 : view(x0, :, 1)
    end
    c = one(real(eltype(sample)))

    if applicable(recurrence, sample, sample, sample, c, 1, sample)
        return recurrence
    end
    if applicable(recurrence, sample, sample, sample, c, sample)
        return (b1, b2, b3, c_, _k, X) -> recurrence(b1, b2, b3, c_, X)
    end

    throw(ArgumentError("clenshaw_recurrence must accept (b1,b2,b3,c,X) or (b1,b2,b3,c,k,X)."))
end
