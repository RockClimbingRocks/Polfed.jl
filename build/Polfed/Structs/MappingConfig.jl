"""
    MappingConfig(; kwargs...)

Configuration for mapping, rescaling, and parallelization.
"""
mutable struct MappingConfig
    parallel_strategy::Union{Parallelization,Nothing}
    optimize_mapping::Bool
    f!_rescaled::Union{Nothing, Function}
    clenshaw_recurrence::Union{Nothing, Function}
    clenshaw_finalsum::Union{Nothing, Function}
    Emin::Union{Real, Nothing}
    Emax::Union{Real, Nothing}

    """Build a `MappingConfig` from keyword arguments."""
    function MappingConfig(;
        parallel_strategy::Union{Parallelization,Nothing} = PolfedDefaults.parallel_strategy,
        optimize_mapping::Bool                          = PolfedDefaults.optimize_mapping,
        f!_rescaled::Union{Nothing, Function}           = nothing,
        clenshaw_recurrence::Union{Nothing, Function}   = nothing,
        clenshaw_finalsum::Union{Nothing, Function}     = nothing,
        Emin::Union{Real, Nothing}                      = nothing,
        Emax::Union{Real, Nothing}                      = nothing,
    )
        new(parallel_strategy, optimize_mapping, f!_rescaled, clenshaw_recurrence, clenshaw_finalsum, Emin, Emax)
    end
end
