
"""
    definetransformation!(transform_plan::TransformPlan, mapping_plan::MappingPlan, x0::AbstractVecOrMat{E}, pu::ProcessingUnit) where {E<:Number} -> Function

Construct the transformed in-place operator used by Lanczos.

The returned callback has signature `f!_transformed(Y, X)` and applies the
resolved Clenshaw polynomial filter based on the current transform/mapping
plans.
"""
function definetransformation!(
    transform_plan::TransformPlan,
    mapping_plan::MappingPlan,
    x0::AbstractVecOrMat{E},
    pu::ProcessingUnit,
) where {E<:Number}

    @unpack coefficients, polynomialtype, normalization, target, order = transform_plan
    hilbertspacedim = size(x0,1)

    transform = Clenshaw(
        polynomialtype,
        n -> coefficients(target, n),
        order,
        mapping_plan.f!_rescaled,
        hilbertspacedim,
        E,
    )
    norm_ = 1 / transform(target) * normalization
    coefficients_normalized(n::Int) = coefficients(target, n) * norm_

    clenshawtransform = define_clenshawtransformation(
        transform_plan,
        mapping_plan,
        coefficients_normalized,
        hilbertspacedim,
        E,
        x0,
    )
    b_storage = get_b_storage(mapping_plan.parallel_strategy, pu, x0)

    f!_transformed = (Y::AbstractVecOrMat{<:Number}, X::AbstractVecOrMat{<:Number}) -> begin
        clenshaw(clenshawtransform, Y, X, b_storage, pu, mapping_plan.parallel_strategy)
        nothing
    end

    return f!_transformed
end


"""
    normalize_clenshaw_recurrence(recurrence::Function, x0::AbstractVecOrMat, parallel_strategy::Parallelization) -> Function

Normalize user-provided Clenshaw recurrence callbacks to the internal
six-argument signature `(b1, b2, b3, c, k, X)`.

Accepted user signatures:
- `(b1, b2, b3, c, k, X)`
- `(b1, b2, b3, c, X)`
"""
function normalize_clenshaw_recurrence(
    recurrence::Function,
    x0::AbstractVecOrMat,
    parallel_strategy::Parallelization,
)
    sample = if parallel_strategy isa NoParallel
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


"""
    define_clenshawtransformation(transform_plan::TransformPlan, mapping_plan::MappingPlan, coefficients_normalized::Function, hilbertspacedim::Int, E::Type, x0::AbstractVecOrMat) -> Clenshaw

Create a [`Clenshaw`](@ref) object using either:
- user-provided recurrence/final-sum callbacks, or
- mapping-derived defaults.
"""
function define_clenshawtransformation(
    transform_plan::TransformPlan,
    mapping_plan::MappingPlan,
    coefficients_normalized::Function,
    hilbertspacedim::Int,
    E::Type,
    x0::AbstractVecOrMat,
)

    is_clenshaw_recurrence_set   = !isnothing(mapping_plan.clenshaw_recurrence)
    is_clenshaw_finalsum_set     = !isnothing(mapping_plan.clenshaw_finalsum)

    if is_clenshaw_recurrence_set && is_clenshaw_finalsum_set
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_INFO_LEVEL,
            "Using user-provided Clenshaw recurrence and final sum.",
        )
        return Clenshaw(
            transform_plan.polynomialtype,
            coefficients_normalized,
            transform_plan.order,
            normalize_clenshaw_recurrence(mapping_plan.clenshaw_recurrence, x0, mapping_plan.parallel_strategy),
            mapping_plan.clenshaw_finalsum,
            hilbertspacedim,
            E,
        )
    elseif is_clenshaw_recurrence_set && !is_clenshaw_finalsum_set
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_INFO_LEVEL,
            "Using user-provided Clenshaw recurrence; final sum derived from mapping.",
        )
        _, finalsum_from_mapping = ClenshawMapping.build_clenshaw_kernels(
            mapping_plan.f!_rescaled,
            transform_plan.polynomialtype,
        )
        return Clenshaw(
            transform_plan.polynomialtype,
            coefficients_normalized,
            transform_plan.order,
            normalize_clenshaw_recurrence(mapping_plan.clenshaw_recurrence, x0, mapping_plan.parallel_strategy),
            finalsum_from_mapping,
            hilbertspacedim,
            E,
        )
    elseif !is_clenshaw_recurrence_set && is_clenshaw_finalsum_set
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_INFO_LEVEL,
            "Using user-provided Clenshaw final sum; recurrence derived from mapping.",
        )
        recurrence_from_mapping, _ = ClenshawMapping.build_clenshaw_kernels(
            mapping_plan.f!_rescaled,
            transform_plan.polynomialtype,
        )
        return Clenshaw(
            transform_plan.polynomialtype,
            coefficients_normalized,
            transform_plan.order,
            recurrence_from_mapping,
            mapping_plan.clenshaw_finalsum,
            hilbertspacedim,
            E,
        )
    end

    PolfedDefaults.polfed_log(
        PolfedDefaults.POLFED_INFO_LEVEL,
        "Using mapping-derived Clenshaw recurrence and final sum.",
    )
    return Clenshaw(
        transform_plan.polynomialtype,
        coefficients_normalized,
        transform_plan.order,
        mapping_plan.f!_rescaled,
        hilbertspacedim,
        E,
    )
end
