
include("Bisection.jl")
include("GetBounds.jl")
include("GetOrderOfExpansion.jl")
include("ParallelizationClenshaw.jl")
include("ParallelizationMultiplication.jl")
include("DefineTransformation.jl")
# include("MultiplicationTypeClenshaw.jl")


"""
    getspectraltransform!(dos::DoSConfigFull, transform_plan::TransformPlan, mapping_plan::MappingPlan, fact::FactorizationConfigFull, pu::ProcessingUnit) -> Function

Build the transformed operator used by POLFED filtering.

This function resolves transform bounds/order and then constructs the concrete
Clenshaw-based transformed mapping callback.
"""
function getspectraltransform!(
    dos::DoSConfigFull,
    transform_plan::TransformPlan,
    mapping_plan::MappingPlan,
    fact::FactorizationConfigFull,
    pu::ProcessingUnit,
)

    getbounds!(transform_plan, mapping_plan, dos.ρ, :mean)
    getorderofexpansion!(transform_plan)

    return definetransformation!(transform_plan, mapping_plan, fact.x0, pu)
end
