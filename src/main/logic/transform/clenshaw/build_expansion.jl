# Build the operator ClenshawExpansion for the transform: the filter polynomial of the
# rescaled Hamiltonian, evaluated by Clenshaw recurrence over `mp.f!_rescaled`. This file
# is purely about *which* polynomial and *which* Clenshaw kernels — the block-splitting
# (workspace + parallel apply) lives in transform/parallel/. All resolution (target,
# bounds, order, validation) already happened upstream in target/ and filter/, so the
# filter arriving here is fully resolved (`target` and `order` set).

"""
    _build_expansion(family, order, coefficients, mp, x0) -> evaluator

Build the operator [`ClenshawExpansion`](@ref) for the transform, selecting the Clenshaw
kernels from the [`MappingPlan`](@ref):

- neither `mp.clenshaw_recurrence` nor `mp.clenshaw_finalsum` set (the usual case) → the
  evaluator for `mp.f!_rescaled` ([`_operator_expansion`](@ref)),
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
- `x0::AbstractVecOrMat`: probe used to detect a user recurrence's arity.

# Returns
An evaluator callable as `expansion(Y, X, b)` — a [`ClenshawExpansion`](@ref) or one of the
fused forms chosen by [`_operator_expansion`](@ref).
"""
function _build_expansion(family::Symbol, order::Int, coefficients,
                          mp::MappingPlan, x0::AbstractVecOrMat)
    has_recurrence = !isnothing(mp.clenshaw_recurrence)
    has_finalsum   = !isnothing(mp.clenshaw_finalsum)

    if !has_recurrence && !has_finalsum
        return _operator_expansion(family, coefficients, order, mp.f!_rescaled)
    end

    recurrence = if has_recurrence
        normalize_clenshaw_recurrence(mp.clenshaw_recurrence, x0, mp.parallel_strategy,
                                      mp.sys.layout)
    else
        first(build_clenshaw_kernels(mp.f!_rescaled, family))
    end
    finalsum = has_finalsum ? mp.clenshaw_finalsum : last(build_clenshaw_kernels(mp.f!_rescaled, family))

    return ClenshawExpansion(family, coefficients, order, recurrence, finalsum)
end

"""
    _operator_expansion(family, coefficients, order, op) -> evaluator

The Clenshaw evaluator for the operator `op`, callable as `e(Y, X, b)`:

- an [`_AccumulatingOperator`](@ref) (generic matrix path, CPU or GPU) → [`AxpbyClenshaw`](@ref):
  every step is one accumulating product plus one `axpy` — on MKL's threads on the CPU, as two
  device calls (CUSPARSE + CUBLAS) on the GPU;
- [`RangeKernelOperator`](@ref) (optimized kernel with a range form) → [`RangeClenshaw`](@ref):
  the whole recurrence in one threaded region;
- anything else → [`ClenshawExpansion`](@ref): operator call, then a threaded combine, per step.

The first two never switch thread pool inside a step.
"""
_operator_expansion(family::Symbol, coefficients, order::Int, op) =
    ClenshawExpansion(family, coefficients, order, op)

_operator_expansion(family::Symbol, coefficients, order::Int, op::_AccumulatingOperator) =
    AxpbyClenshaw(family, coefficients, order,
                  (Y, X, α, β) -> _axpby!(op, Y, X, α, β),
                  (Y, c, X) -> _add!(op, Y, c, X))

_operator_expansion(family::Symbol, coefficients, order::Int, op::RangeKernelOperator) =
    RangeClenshaw(family, coefficients, order, op.range!, op.units; alignment = op.alignment)

"""
    normalize_clenshaw_recurrence(recurrence, x0, strategy::Parallelization) -> Function

Adapt a user Clenshaw recurrence to the internal six-argument signature
`(b1, b2, b3, c, k, X)`. Accepts either that form or the five-argument
`(b1, b2, b3, c, X)` (the step index `k` is then ignored). The `strategy` selects the
sample used to probe which form the callback accepts (a single column view for the
per-vector strategies, the whole `x0` for [`BlockWise`](@ref)). The single-vector sample is
taken through [`_vector_slice`](@ref), so it is a `D`-length vector under either layout.

# Throws
- `ArgumentError`: if `recurrence` accepts neither signature.
"""
function normalize_clenshaw_recurrence(recurrence, x0::AbstractVecOrMat, strategy::Parallelization,
                                       layout::Layout = ColumnLayout())
    sample = strategy isa BlockWise ? x0 : _vector_slice(layout, x0, 1)
    c = one(real(eltype(sample)))

    if applicable(recurrence, sample, sample, sample, c, 1, sample)
        return recurrence
    end
    if applicable(recurrence, sample, sample, sample, c, sample)
        return (b1, b2, b3, c_, _k, X) -> recurrence(b1, b2, b3, c_, X)
    end

    throw(ArgumentError("clenshaw_recurrence must accept (b1,b2,b3,c,X) or (b1,b2,b3,c,k,X)."))
end
