# Mapping slice: rescale the operator to `[-1, 1]` (required by the Chebyshev filter)
# and resolve how that application is parallelized. This file is the folder aggregator
# — it pulls in the parallelization strategy types (chosen here, consumed downstream)

include("parallel.jl")       # Parallelization types + resolve/warn (MappingConfig field + build)
include("row_operators.jl")  # row-layout operator builders for the matrix entrypoint (polfed(H, …))

"""
    MappingConfig(; parallel_strategy=nothing, optimize_mapping=Defaults.polfed.optimize_mapping,
                    f!_rescaled=nothing, clenshaw_recurrence=nothing, clenshaw_finalsum=nothing,
                    Emin=nothing, Emax=nothing)

User-facing configuration for POLFED's **mapping** stage — how the operator is
rescaled to `[-1, 1]` and how that application is parallelized/optimized.

The mapping rescales the operator to `H̃ = (H - b)/a` with `a = (Emax-Emin)/2`,
`b = (Emax+Emin)/2`, so its spectrum lands in `[-1, 1]` (required by the Chebyshev
filter). [`build_mapping_plan`](@ref) resolves this into a [`MappingPlan`](@ref):
it estimates any missing spectral bound with a short [`lanczos_extrema`](@ref) probe
and builds the `f!_rescaled` closure.

# Fields
- `optimize_mapping`: build optimized value-grouped kernels from a matrix input —
  `false`, `true`/`:auto`, a kernel name, or a [`MappingKernel`](@ref)
  (for structured Hamiltonians). Default [`Defaults.polfed.optimize_mapping`](@ref) (`false`).
- `f!_rescaled::Union{Nothing,Function}`: user-supplied rescaled operator `H̃·X`; if
  `nothing`, POLFED builds it from `f!` and `a`/`b`.
- `clenshaw_recurrence`, `clenshaw_finalsum::Union{Nothing,Function}`: optional custom
  Clenshaw kernels (consumed by the transform stage); `nothing` uses the defaults.
- `parallel_strategy::Union{Parallelization,Nothing}`: how the operator application is
  parallelized. `nothing` (default) means auto — resolved per processing unit
  (CPU → [`VectorWise`](@ref), GPU → [`BlockWise`](@ref)).
- `Emin`, `Emax::Union{Real,Nothing}`: explicit spectral bounds. Either left `nothing`
  is estimated by a short Lanczos probe.

# Example
```julia
MappingConfig()                          # auto bounds + auto parallelization
MappingConfig(Emin = -12.0, Emax = 12.0) # skip the bound-estimation probe
```

# Notes
Immutable. The resolved parallelization strategy lives on the [`MappingPlan`](@ref),
not here (the config is not mutated). Supplying `Emin`/`Emax` skips the estimation
probe — useful when you already know the spectral edges.
"""
struct MappingConfig
    optimize_mapping::Union{Bool,Symbol,NamedTuple,MappingKernel}
    f!_rescaled::Union{Nothing,Function}
    clenshaw_recurrence::Union{Nothing,Function}
    clenshaw_finalsum::Union{Nothing,Function}
    parallel_strategy::Union{Parallelization,Nothing}
    Emin::Union{Real,Nothing}
    Emax::Union{Real,Nothing}

    function MappingConfig(;
        optimize_mapping::Union{Bool,Symbol,NamedTuple,MappingKernel}   = Defaults.polfed.optimize_mapping,
        f!_rescaled::Union{Nothing,Function}              = nothing,
        clenshaw_recurrence::Union{Nothing,Function}      = nothing,
        clenshaw_finalsum::Union{Nothing,Function}        = nothing,
        parallel_strategy::Union{Parallelization,Nothing} = nothing,
        Emin::Union{Real,Nothing}                         = nothing,
        Emax::Union{Real,Nothing}                         = nothing,
    )
        new(optimize_mapping, f!_rescaled,
            clenshaw_recurrence, clenshaw_finalsum, parallel_strategy, Emin, Emax)
    end
end

"""
    MappingPlan{T,Fpar,Fmap,Frescaled,Fcrr,Fcfs}

Resolved mapping plan produced by [`build_mapping_plan`](@ref) from a
[`MappingConfig`](@ref). Holds the concrete rescaling and the callbacks the rest of
POLFED runs on.

Fully parametrized so the hot path (`f!_rescaled` applied each Clenshaw step,
dispatch on `parallel_strategy`) is type-stable: the strategy, the operator
callbacks, and the scalar bounds all carry concrete types.

# Fields
- `parallel_strategy::Fpar`: resolved [`Parallelization`](@ref) strategy.
- `optimize_mapping::Union{Bool,Symbol,NamedTuple,MappingKernel}`: the mapping-optimization request carried
  from the config (metadata; the optimized kernels themselves are built in the matrix entrypoint
  and passed in via `f!_rescaled`).
- `f!::Fmap`: the original in-place operator `H·X`.
- `f!_rescaled::Frescaled`: in-place rescaled operator `H̃·X = ((H - b)/a)·X`.
- `clenshaw_recurrence::Fcrr`, `clenshaw_finalsum::Fcfs`: custom Clenshaw kernels or
  `nothing` (carried for the transform stage).
- `Emin::T`, `Emax::T`: spectral bounds (provided or estimated).
- `a::T`, `b::T`: rescaling scale and shift, `a = (Emax-Emin)/2`, `b = (Emax+Emin)/2`.
- `sys::S`: the [`SysInfo`](@ref) problem context (`dim`, `block_size`, `pu`, `x0`,
  element type), so downstream stages take just the plan — accessed as `mp.sys.dim`,
  `mp.sys.pu`, etc.
"""
struct MappingPlan{T<:Real,Fpar<:Parallelization,Fmap,Frescaled,Fcrr,Fcfs,S<:SysInfo}
    parallel_strategy::Fpar
    optimize_mapping::Union{Bool,Symbol,NamedTuple,MappingKernel}
    f!::Fmap
    f!_rescaled::Frescaled
    clenshaw_recurrence::Fcrr
    clenshaw_finalsum::Fcfs
    Emin::T
    Emax::T
    a::T
    b::T
    sys::S
end

"""
    _make_rescaled(f!, a, b) -> Function

Build the in-place rescaled operator `H̃·X = ((H - b)/a)·X` as a closure capturing
`f!`, `a`, `b`: it applies `f!` then `Y ← Y/a - (b/a)·X`.
"""
function _make_rescaled(f!::F, a, b) where {F}
    return @inline (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
        f!(Y, X)
        @. Y *= 1 / a
        @. Y -= (b / a) * X
        return nothing
    end
end

"""
    _resolve_bounds(cfg::MappingConfig, f!, v0) -> (Emin, Emax)

Return the spectral bounds: each of `cfg.Emin`/`cfg.Emax` that is `nothing` is estimated
with a single [`lanczos_extrema`](@ref) probe (`BothEnds`, power-iteration warm-up). When
a probe runs, its convergence is checked and a warning is emitted if both extremes did not
converge (the rescaling would be inaccurate). If both bounds are supplied, no probe runs.
"""
function _resolve_bounds(cfg::MappingConfig, f!, v0::AbstractVector)
    if !isnothing(cfg.Emin) && !isnothing(cfg.Emax)
        polfed_log(POLFED_DEBUG_LEVEL, "Spectral bounds supplied; skipping the Lanczos probe.",
                   Emin = cfg.Emin, Emax = cfg.Emax)
        return (cfg.Emin, cfg.Emax)
    end

    polfed_log(POLFED_DEBUG_LEVEL, "Estimating spectral bounds with a Lanczos probe.",
               estimating = isnothing(cfg.Emin) ? (isnothing(cfg.Emax) ? "Emin+Emax" : "Emin") : "Emax")

    res = lanczos_extrema(f!, v0; return_result = true)
    polfed_log(POLFED_DEBUG_LEVEL, "Bounds probe finished.",
               converged = res.converged, vals = res.vals,
               iterations = res.report.itersneeded, residual = res.report.residual)
    if res.converged < 2
        polfed_log(POLFED_WARN_LEVEL,
                   "Spectral-bounds probe did not converge both extremes; estimated Emin/Emax " *
                   "(and the rescaling) may be inaccurate — increase the probe maxdim or pass " *
                   "explicit Emin/Emax.",
                   converged = res.converged)
    end
    vals = collect(res.vals)
    Emin = isnothing(cfg.Emin) ? first(vals) : cfg.Emin
    Emax = isnothing(cfg.Emax) ? last(vals)  : cfg.Emax
    return Emin, Emax
end

"""
    build_mapping_plan(cfg::MappingConfig, f!, x0, pu=CPU()/GPU() from x0;
                       layout=ColumnLayout()) -> MappingPlan

Resolve a [`MappingConfig`](@ref) into a concrete [`MappingPlan`](@ref):

1. resolve + validate the parallelization strategy (per processing unit),
2. estimate any missing spectral bound with a short [`lanczos_extrema`](@ref) probe,
3. compute the rescaling `a = (Emax-Emin)/2`, `b = (Emax+Emin)/2`,
4. build the in-place `f!_rescaled` closure (`H̃ = (H - b)/a`) unless the user supplied one.

`x0` is the POLFED starting vector/block; its first **Krylov vector** seeds the
bound-estimation probe — which axis that is depends on `layout`, so the slice is taken
through [`_vector_slice`](@ref) rather than assuming a column. The processing unit `pu`
defaults to the device `x0` lives on ([`GPU`](@ref) if `x0` is a GPU array, otherwise
[`CPU`](@ref)), and can be passed explicitly to override.

`layout` is how the Krylov block is stored ([`ColumnLayout`](@ref) `D × bs` or
[`RowLayout`](@ref) `bs × D`). It is recorded on the plan's [`SysInfo`](@ref) and is the
single source of orientation for every downstream stage that slices a vector out of a block.
"""
function build_mapping_plan(cfg::MappingConfig, f!::Fmap, x0::AbstractVecOrMat,
                            pu::ProcessingUnit = is_gpu_array(x0) ? GPU() : CPU();
                            layout::Layout = ColumnLayout()) where {Fmap}

    strategy = resolve_parallelization(cfg.parallel_strategy, pu, blocksize(layout, x0),
                                       cfg.optimize_mapping)
    warn_parallelization(x0, strategy, pu, layout)

    # `vec`: the probe seed is a plain length-dim vector, but a row-layout slice is `1 × dim`.
    v0 = pu.Vector(vec(_vector_slice(layout, x0, 1)))

    Emin, Emax = _resolve_bounds(cfg, f!, v0)

    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2
    T = typeof(a)

    f!_rescaled = isnothing(cfg.f!_rescaled) ? _make_rescaled(f!, a, b) : cfg.f!_rescaled

    sys = SysInfo(x0, pu, layout)   # dim/block_size/pu/eltype/layout + x0 (by reference); mp.sys

    polfed_log(POLFED_INFO_LEVEL, "Mapping resolved: H̃ = (H - b)/a rescales the spectrum to [-1, 1].",
               Emin = Emin, Emax = Emax, a = a, b = b,
               dim = sys.dim, block_size = sys.block_size,
               eltype = eltype(sys.x0), processing_unit = typeof(pu),
               layout = typeof(layout),
               parallel = typeof(strategy), width = Emax - Emin,
               f!_rescaled = isnothing(cfg.f!_rescaled) ? "derived" : "user-provided")

    return MappingPlan(
        strategy, cfg.optimize_mapping, f!, f!_rescaled,
        cfg.clenshaw_recurrence, cfg.clenshaw_finalsum,
        T(Emin), T(Emax), a, b, sys,
    )
end
