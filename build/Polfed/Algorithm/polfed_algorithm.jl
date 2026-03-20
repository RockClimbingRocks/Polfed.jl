"""
    polfed_algorithm(transform_plan::TransformPlan, mapping_plan::MappingPlan, fact_config::FactorizationConfigFull, dos::DoSConfigFull, pu::ProcessingUnit)

Execute the core POLFED pipeline after all plans/configs are resolved.

Steps:
1. Estimate DoS and build `dos.ρ`.
2. Construct transformed operator via spectral filtering.
3. Run Lanczos/block-Lanczos on the transformed operator.

# Returns
- `(vals, vecs, fact_report)` where `fact_report` is a
  [`FactorizationReport`](@ref).
"""
function polfed_algorithm(
    transform_plan::TransformPlan,
    mapping_plan::MappingPlan,
    fact_config::FactorizationConfigFull,
    dos::DoSConfigFull,
    pu::ProcessingUnit,
)
    getdos!(dos, mapping_plan, fact_config, pu)
    f!_transformed = getspectraltransform!(dos, transform_plan, mapping_plan, fact_config, pu)

    vals, vecs, fact_report = lanczos(
        f!_transformed, fact_config.x0, transform_plan.howmany;
        rot         = fact_config.rot,
        basistype   = fact_config.basistype, 
        maxdim      = fact_config.maxdim, 
        tol         = fact_config.tol, 
        eigentol    = fact_config.eigentol, 
        which       = fact_config.which,
        mapvals     = mapping_plan.f!
    ) 

    return vals, vecs, fact_report
end
