
"""
    MappingPlan

Resolved mapping plan used during a POLFED run.

This struct is created from [`MappingConfig`](@ref) and stores the concrete
mapping callbacks, selected parallelization strategy, and rescaling constants
derived from `Emin`/`Emax`.
"""
mutable struct MappingPlan{Fmap,Frescaled,Fcrr,Fcfs}
    parallel_strategy::Parallelization
    f!::Fmap
    f!_rescaled::Frescaled
    clenshaw_recurrence::Union{Nothing, Fcrr}
    clenshaw_finalsum::Union{Nothing, Fcfs}
    Emin::Real
    Emax::Real
    a::Real
    b::Real
end

"""
    TransformPlan

Resolved spectral-transform plan used during a POLFED run.

This struct is created from [`TransformConfig`](@ref) and stores the concrete
target specification, bounds/order state, and normalized transform settings.
"""
mutable struct TransformPlan{Fcoef,Tspec}
    coefficients::Fcoef
    normalization::Real
    polynomialtype::Symbol
    cutoff::Real
    target_spec::Tspec
    target::Union{Real, Nothing}
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    howmany::Integer
    order::Union{Int, Nothing}
    order_safety_factor::Real
end

"""
    should_warn() -> Bool

Return `true` when POLFED warning-level logging is enabled.
"""
@inline function should_warn()
    return PolfedDefaults.verbosity[] >= PolfedDefaults.POLFED_WARN_LEVEL
end

"""
    resolve_parallelization(mapping::MappingConfig, pu::ProcessingUnit) -> Parallelization

Resolve the effective parallelization strategy for a run.

If `mapping.parallel_strategy` is `nothing`, the default is:
- [`MulColsParallel`](@ref) on CPU
- [`NoParallel`](@ref) on GPU
"""
function resolve_parallelization(mapping::MappingConfig, pu::ProcessingUnit)
    if mapping.parallel_strategy === nothing
        return isa(pu, CPU) ? MulColsParallel() : NoParallel()
    end
    return mapping.parallel_strategy
end

"""
    warn_parallelization(x0, parallel_strategy, pu) -> nothing

Emit runtime warnings for potentially suboptimal parallelization/device choices.

Checks include:
- thread/column mismatch for [`MulColsParallel`](@ref),
- CUDA availability while running on CPU,
- multi-GPU systems where only the default device is used.
"""
function warn_parallelization(x0::AbstractVecOrMat, parallel_strategy::Parallelization, pu::ProcessingUnit)
    if should_warn()
        cuda_devices = 0
        if CUDA_AVAILABLE
            try
                cuda_devices = CUDA.functional() ? CUDA.device_count() : 0
            catch
                cuda_devices = 0
            end
        end

        if parallel_strategy isa MulColsParallel && x0 isa AbstractMatrix
            ncols = size(x0, 2)
            nthreads = Threads.nthreads()
            ncols != nthreads && @warn "MulColsParallel with column count different from thread count." columns=ncols threads=nthreads
        end

        if cuda_devices > 0 && isa(pu, CPU)
            @warn "CUDA is available but input is on CPU; GPU not used."
        end

        if cuda_devices > 1 && isa(pu, GPU)
            @warn "Multiple CUDA devices available; using default device only." devices=cuda_devices
        end
    end

    return nothing
end

"""
    build_mapping_plan(mapping::MappingConfig, f!, x0, pu) -> MappingPlan

Build a fully-resolved mapping plan.

# Arguments
- `mapping::MappingConfig`: User mapping configuration.
- `f!`: Original in-place operator callback.
- `x0::AbstractVecOrMat`: Initial Lanczos vector/block.
- `pu::ProcessingUnit`: Selected processing unit.

# Returns
- `MappingPlan`: Contains parallel strategy, original/rescaled mapping
  callbacks, optional Clenshaw overrides, and rescaling constants.
"""
function build_mapping_plan(mapping::MappingConfig, f!::Fmap, x0::AbstractVecOrMat, pu::ProcessingUnit) where {Fmap}
    parallel_strategy = resolve_parallelization(mapping, pu)
    mapping.parallel_strategy === nothing && (mapping.parallel_strategy = parallel_strategy)
    warn_parallelization(x0, parallel_strategy, pu)

    v0_source = x0 isa AbstractVector ? x0 : @view x0[:, 1]
    v0 = pu.Vector(v0_source)

    if isnothing(mapping.Emin)
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_DEBUG_LEVEL,
            "Estimating minimum eigenvalue (Emin) with Lanczos.",
            which=:SR,
            maxdim=1000,
        )
        Emin = first(collect(lanczos(f!, v0, 1; which=:SR, maxdim=1000)[1]))
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_DEBUG_LEVEL,
            "Finished Emin estimation.",
            Emin=Emin,
        )
    else
        Emin = mapping.Emin
    end

    if isnothing(mapping.Emax)
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_DEBUG_LEVEL,
            "Estimating maximum eigenvalue (Emax) with Lanczos.",
            which=:LR,
            maxdim=1000,
        )
        Emax = last(collect(lanczos(f!, v0, 1; which=:LR, maxdim=1000)[1]))
        PolfedDefaults.polfed_log(
            PolfedDefaults.POLFED_DEBUG_LEVEL,
            "Finished Emax estimation.",
            Emax=Emax,
        )
    else
        Emax = mapping.Emax
    end

    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    f!_rescaled = !isnothing(mapping.f!_rescaled) ? mapping.f!_rescaled : @inline (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
        f!(Y, X)
        @. Y *= 1 / a
        @. Y -= (b / a) * X
    end

    return MappingPlan{Fmap, typeof(f!_rescaled), typeof(mapping.clenshaw_recurrence), typeof(mapping.clenshaw_finalsum)}(
        parallel_strategy,
        f!,
        f!_rescaled,
        mapping.clenshaw_recurrence,
        mapping.clenshaw_finalsum,
        Emin,
        Emax,
        a,
        b,
    )
end

"""
    build_transform_plan(transform::TransformConfig, mapping_plan::MappingPlan, howmany, target_spec_input) -> TransformPlan

Build a transform plan in rescaled coordinates.

This resolves:
- target specification (`Symbol`/tuple/number),
- optional interval bounds (`left`, `right`) into rescaled units,
- fixed target values for `TargetAbsolute`/`TargetRescaled`/`TargetMiddle`.
"""
function build_transform_plan(
    transform::TransformConfig,
    mapping_plan::MappingPlan,
    howmany::Integer,
    target_spec_input
)
    target_spec = normalize_target_spec(target_spec_input)

    left_rescale  = isnothing(transform.left)  ? nothing : (transform.left - mapping_plan.b) / mapping_plan.a
    right_rescale = isnothing(transform.right) ? nothing : (transform.right - mapping_plan.b) / mapping_plan.a

    target_rescale = nothing
    if target_spec isa TargetAbsolute
        target_rescale = (target_spec.value - mapping_plan.b) / mapping_plan.a
    elseif target_spec isa TargetRescaled
        target_rescale = target_spec.value
    elseif target_spec isa TargetMiddle
        target_rescale = 0.0
    end

    return TransformPlan{typeof(transform.coefficients), typeof(target_spec)}(
        transform.coefficients,
        transform.normalization,
        transform.polynomialtype,
        transform.cutoff,
        target_spec,
        target_rescale,
        left_rescale,
        right_rescale,
        howmany,
        transform.order,
        transform.order_safety_factor,
    )
end

"""
    findmaximaldensity(ρ::Function) -> Real

Estimate the location of maximal density of states in `[-0.99, 0.99]`.
"""
function findmaximaldensity(ρ::Function)
    x = LinRange(-0.99, 0.99, 5000)
    _, i = findmax(xi -> ρ(xi), x)

    PolfedDefaults.verbosity[] >= PolfedDefaults.POLFED_INFO_LEVEL && @info "Targeting maximal density of states." energy=x[i]
    return x[i]
end

"""
    resolve_target!(transform_plan::TransformPlan, mapping_plan::MappingPlan, ρ::Function) -> nothing

Resolve `transform_plan.target` from `target_spec` when it is not already set.

Supported target specs include:
- `TargetMaxDoS`, `TargetMiddle`, `TargetOffset`, `TargetRescaled`,
  and `TargetAbsolute`.
"""
function resolve_target!(transform_plan::TransformPlan, mapping_plan::MappingPlan, ρ::Function)
    !isnothing(transform_plan.target) && return

    spec = transform_plan.target_spec
    if spec isa TargetMaxDoS
        transform_plan.target = findmaximaldensity(ρ)
    elseif spec isa TargetOffset
        peak = findmaximaldensity(ρ)
        eta = spec.frac
        transform_plan.target = eta >= 0 ?
            peak + eta * (1.0 - peak) :
            peak + eta * (1.0 + peak)
    elseif spec isa TargetMiddle
        transform_plan.target = 0.0
    elseif spec isa TargetRescaled
        transform_plan.target = spec.value
    elseif spec isa TargetAbsolute
        transform_plan.target = (spec.value - mapping_plan.b) / mapping_plan.a
    else
        throw(ArgumentError("Unsupported target specification: $spec"))
    end
end
