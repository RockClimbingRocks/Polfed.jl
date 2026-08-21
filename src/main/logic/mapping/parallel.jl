# Parallelization strategy for the mapping/transform matvec. The mapping stage is
# where the strategy is *chosen* (MappingConfig.parallel_strategy → resolved onto the
# MappingPlan), so the strategy types and their resolution/validation live here. The
# strategy-dispatched matvec itself (b-storage + parallel Clenshaw) is on the
# *consumption* side, with the transform/optimization slice — not here.

"""
    Parallelization

Abstract supertype for POLFED's mapping/transform parallelization strategies. Used
for multiple dispatch: how the (block) operator application is parallelized is
selected by the concrete strategy type.

Concrete strategies: [`BlockWise`](@ref), [`VectorWise`](@ref).

"""
abstract type Parallelization end

"""
    BlockWise(; groups = nothing)

Parallelism **inside** a single Clenshaw recurrence step: the Krylov block is handed to the
operator in one call and split across threads there, so one parallel region is opened per step.

`groups` is how many Krylov-vector groups the block is cut into; each group is served by
`nthreads() ÷ groups` threads splitting its basis-state range. `groups = 1` is the pure
basis-state split — the only parallelism available at block size 1. `nothing` resolves it from
the block size, the thread count and the cache geometry.

On a [`GPU`](@ref) this is always the strategy: the device kernels own the decomposition.

# Fields
- `groups::Union{Int,Nothing}`: vector groups, or `nothing` for automatic resolution.

!!! note
    `groups` is not read yet — resolution arrives with the block decomposition. Until then
    every `BlockWise` behaves as the whole-block call this replaced.
"""
struct BlockWise <: Parallelization
    groups::Union{Int,Nothing}
end
BlockWise(; groups::Union{Int,Nothing} = nothing) = BlockWise(groups)

"""
    VectorWise()

Parallelism **around** the Clenshaw call: the recurrence runs once per Krylov vector, each with
its own buffer triplet. Nothing is shared between threads, so no per-step barrier is needed and
one parallel region covers the whole `K`-step loop.

Currently splits the block by **columns**; layout-awareness (rows under [`RowLayout`](@ref))
is not implemented yet. Gives no parallelism at block size 1 — use [`BlockWise`](@ref) there.
"""
struct VectorWise <: Parallelization end



# ── resolution & validation ───────────────────────────────────────────────────

"""
    resolve_parallelization(strategy, pu::ProcessingUnit) -> Parallelization

Resolve the effective parallelization strategy for a run.

- On a [`GPU`](@ref) processing unit, always [`BlockWise`](@ref) (the device kernels
  handle the decomposition themselves).
- On [`CPU`](@ref), `nothing` (auto) becomes [`VectorWise`](@ref); an explicit
  strategy is returned unchanged.

The default is intentionally *computed* (it depends on the processing unit) rather
than stored as a constant — there is no parallelization entry in `Defaults`.
"""
function resolve_parallelization(strategy::Union{Parallelization,Nothing}, pu::ProcessingUnit)
    !isnothing(strategy) && return strategy
    isa(pu, GPU) && return BlockWise()
    isa(pu, CPU) && return VectorWise()
    throw(ArgumentError("Unrecognized processing unit $pu"))
end

"""
    should_warn() -> Bool

Whether POLFED warning-level logging is currently enabled (see [`should_log`](@ref)).
"""
should_warn() = should_log(POLFED_WARN_LEVEL)

"""
    warn_parallelization(x0, strategy, pu) -> nothing

Emit warnings for potentially suboptimal parallelization/device choices:

- a [`VectorWise`](@ref) block whose column count differs from the thread count,
- CUDA available while running on the [`CPU`](@ref),
- multiple CUDA devices while only the default device is used.

No-op when warning-level logging is disabled.
"""
function warn_parallelization(x0::AbstractVecOrMat, strategy::Parallelization, pu::ProcessingUnit)
    should_warn() || return nothing

    cuda_devices = 0
    if cuda_available()
        try
            cuda_devices = gpu_functional() ? gpu_device_count() : 0
        catch
            cuda_devices = 0
        end
    end

    if strategy isa VectorWise && x0 isa AbstractMatrix
        ncols = size(x0, 2)
        nthreads = Threads.nthreads()
        ncols != nthreads && polfed_log(POLFED_WARN_LEVEL,
            "VectorWise with column count different from thread count.",
            columns = ncols, threads = nthreads)
    end
    if cuda_devices > 0 && isa(pu, CPU)
        polfed_log(POLFED_WARN_LEVEL, "CUDA is available but input is on CPU; GPU not used.")
    end
    if cuda_devices > 1 && isa(pu, GPU)
        polfed_log(POLFED_WARN_LEVEL, "Multiple CUDA devices available; using default device only.",
                   devices = cuda_devices)
    end
    return nothing
end
