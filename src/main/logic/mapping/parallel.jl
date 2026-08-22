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

    # Validated at CONSTRUCTION, not where the decomposition eventually reads it: `groups` is a
    # user-facing knob, and `groups = 0` should fail at the call site that wrote it rather than
    # much later inside a resolution the caller never named.
    function BlockWise(groups::Union{Int,Nothing})
        isnothing(groups) || groups >= 1 || throw(ArgumentError(
            "BlockWise: groups must be >= 1 (or nothing for automatic resolution), got $groups."))
        return new(groups)
    end
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



"""
    resolve_parallelization(strategy, pu::ProcessingUnit, block_size::Int,
                            nthreads::Int = Threads.nthreads()) -> Parallelization

Resolve the effective parallelization strategy for a run. An explicit `strategy` is returned
unchanged; `nothing` (auto) is decided from the block size and the thread count.

- [`GPU`](@ref) → always [`BlockWise`](@ref): the device kernels own the decomposition.
- `block_size == 1` → [`BlockWise`](@ref). [`VectorWise`](@ref) is one Clenshaw per Krylov
  vector, so at a single vector it is serial; the basis-state split is the only parallelism.
- `nthreads <= block_size` → [`VectorWise`](@ref), **when it can run** (see below). Every
  thread gets at least one whole vector, so no buffer is ever handed between cores.
- otherwise (`nthreads > block_size`) → [`BlockWise`](@ref). `VectorWise` cannot use more
  than `block_size` threads, so it would leave `nthreads - block_size` of them idle.

!!! warning "VectorWise is currently unusable above one thread"
    [`VectorWise`](@ref) opens a parallel region per Krylov vector, and the Clenshaw combine
    (`_threaded_combine!`) opens a `Threads.@threads :static` region inside it whenever
    `nthreads() > 1`. Julia rejects `:static` regions running concurrently, so such a run raises
    "`@threads :static` cannot be used concurrently or nested". Until the recurrence can be built
    serial for a caller that owns the threads, auto-resolution will not hand out `VectorWise`
    above one thread, and an explicit request is coerced with a warning — an automatic choice
    must never be one that cannot run.

Divisibility does NOT enter here. `BlockWise` always has a valid decomposition — worst case
one group, the pure basis-state split — so `nthreads % block_size != 0` needs no special
case; it only affects how many groups `BlockWise` resolves internally.

The default is intentionally *computed* rather than stored as a constant — there is no
parallelization entry in `Defaults`.
"""
function resolve_parallelization(strategy::Union{Parallelization,Nothing}, pu::ProcessingUnit,
                                 block_size::Int, nthreads::Int = Threads.nthreads())
    !isnothing(strategy) && return _usable(strategy, nthreads)
    isa(pu, GPU) && return BlockWise()
    isa(pu, CPU) || throw(ArgumentError("Unrecognized processing unit $pu"))
    block_size == 1 && return BlockWise()
    nthreads <= block_size && return _usable(VectorWise(), nthreads)
    return BlockWise()
end

"""
    _usable(strategy::Parallelization, nthreads::Int) -> Parallelization

Downgrade a strategy that cannot run at this thread count, warning when it does.

Only [`VectorWise`](@ref) above one thread is affected, and only because the Clenshaw combine
opens a `Threads.@threads :static` region inside VectorWise's own — see the note on
[`resolve_parallelization`](@ref). This is applied to an EXPLICIT request too: silently running
into "`@threads :static` cannot be used concurrently or nested" from deep inside a task is worse
than being told the strategy was swapped.
"""
function _usable(strategy::Parallelization, nthreads::Int)
    (strategy isa VectorWise && nthreads > 1) || return strategy
    polfed_log(POLFED_WARN_LEVEL,
               "VectorWise cannot run above one thread — the Clenshaw combine opens its own " *
               "`@threads :static` region inside VectorWise's. Using BlockWise instead.",
               threads = nthreads)
    return BlockWise()
end


"""
    should_warn() -> Bool

Whether POLFED warning-level logging is currently enabled (see [`should_log`](@ref)).
"""
should_warn() = should_log(POLFED_WARN_LEVEL)

"""
    warn_parallelization(x0, strategy, pu) -> nothing

Emit warnings for potentially suboptimal parallelization/device choices:

- a [`VectorWise`](@ref) block with FEWER Krylov vectors than threads, which leaves threads
  idle (auto-resolution never produces this — it picks [`BlockWise`](@ref) — so it means the
  strategy was set explicitly),
- CUDA available while running on the [`CPU`](@ref),
- multiple CUDA devices while only the default device is used.

No-op when warning-level logging is disabled.
"""
function warn_parallelization(x0::AbstractVecOrMat, strategy::Parallelization,
                              pu::ProcessingUnit, layout::Layout = ColumnLayout())
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
        nvecs = blocksize(layout, x0)
        nthreads = Threads.nthreads()
        # Only the wasteful direction. `nvecs > nthreads` is the INTENDED VectorWise case —
        # each thread simply takes several whole vectors, and nothing is handed between cores.
        # Counting with `size(x0, 2)` instead of the layout would read the Hilbert dimension
        # as the vector count under RowLayout, and warn on every run.
        nvecs < nthreads && polfed_log(POLFED_WARN_LEVEL,
            "VectorWise has fewer Krylov vectors than threads, so threads sit idle; " *
            "BlockWise would use them.",
            vectors = nvecs, threads = nthreads)
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
