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

Concrete strategies: [`NoParallel`](@ref), [`MulColsParallel`](@ref),
[`MulRowsParallel`](@ref), [`TwoLevelParallel`](@ref).
"""
abstract type Parallelization end

"""
    NoParallel()

Serial execution — no parallelization. Useful for debugging/testing, and when all
parallelism is handled elsewhere (e.g. on the GPU, where the kernels are already
parallel).
"""
mutable struct NoParallel <: Parallelization end

"""
    MulColsParallel(nt_per_col::Int = 1)

Column-parallel block operator application: the independent columns of the Krylov
block are spread across threads (one `H·v` per column). Effective when the block
size is comparable to the thread count; gives no parallelism for a single vector
(block size `= 1`).

# Fields
- `nt_per_col::Int`: threads used *within* each column's matvec (nested intra-column
  parallelism); `1` means pure column parallelism.
"""
mutable struct MulColsParallel <: Parallelization
    nt_per_col::Int
    MulColsParallel(nt_per_col::Int = 1) = new(nt_per_col)
end

"""
    MulRowsParallel(nt_per_row::Int = 1)

Row-parallel block operator application: the operator's output **rows** are chunked
across threads (each thread computes a band of output rows for all block columns).
Unlike [`MulColsParallel`](@ref), this parallelizes **even for a single vector**
(block size `= 1`), and pairs naturally with the row-major block layout
([`RowLayout`](@ref)), where row chunks are contiguous and fast for sparse `H`.

# Fields
- `nt_per_row::Int`: threads used within each row chunk (nested parallelism); `1`
  means pure row-chunk parallelism.

!!! note
    The strategy *type* exists now; the row-chunked matvec it selects is implemented
    in the optimization slice (migrated later). Until then `MulRowsParallel` is a
    dormant placeholder behind the parallelization dispatch.
"""
mutable struct MulRowsParallel <: Parallelization
    nt_per_row::Int
    MulRowsParallel(nt_per_row::Int = 1) = new(nt_per_row)
end

"""
    TwoLevelParallel(nt_per_col::Int)

Two-level parallelization for large problems:
1. **Inter-column** — block columns are distributed across Distributed **worker
   processes** (a `WorkerPool`).
2. **Intra-column** — within each worker, `nt_per_col` threads parallelize the matvec.

The worker pool and a shared Clenshaw buffer are populated at run time by
[`set_workers`](@ref) and torn down by [`remove_workers`](@ref).

# Fields
- `worker_pool::WorkerPool`: worker processes handling column-level tasks (empty
  until `set_workers`).
- `nt_per_col::Int`: threads per worker for intra-column parallelism.
- `clenshaw_shared_buffer::Any`: scratch buffer shared across the Clenshaw recurrence
  on workers (set during the transform; `nothing` until then).
"""
mutable struct TwoLevelParallel <: Parallelization
    worker_pool::WorkerPool
    nt_per_col::Int
    clenshaw_shared_buffer::Any
    TwoLevelParallel(nt_per_col::Int) = new(WorkerPool(), nt_per_col, nothing)
end

# ── resolution & validation ───────────────────────────────────────────────────

"""
    resolve_parallelization(strategy, pu::ProcessingUnit) -> Parallelization

Resolve the effective parallelization strategy for a run.

- On a [`GPU`](@ref) processing unit, always [`NoParallel`](@ref) (the GPU kernels
  handle parallelism themselves).
- On [`CPU`](@ref), `nothing` (auto) becomes [`MulColsParallel`](@ref); an explicit
  strategy is returned unchanged.

The default is intentionally *computed* (it depends on the processing unit) rather
than stored as a constant — there is no parallelization entry in `Defaults`.
"""
function resolve_parallelization(strategy::Union{Parallelization,Nothing}, pu::ProcessingUnit)
    isa(pu, GPU) && return NoParallel()
    isnothing(strategy) && return MulColsParallel()
    return strategy
end

"""
    should_warn() -> Bool

Whether POLFED warning-level logging is currently enabled (see [`should_log`](@ref)).
"""
should_warn() = should_log(POLFED_WARN_LEVEL)

"""
    warn_parallelization(x0, strategy, pu) -> nothing

Emit warnings for potentially suboptimal parallelization/device choices:

- a [`MulColsParallel`](@ref) block whose column count differs from the thread count,
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

    if strategy isa MulColsParallel && x0 isa AbstractMatrix
        ncols = size(x0, 2)
        nthreads = Threads.nthreads()
        ncols != nthreads &&
            @warn "MulColsParallel with column count different from thread count." columns = ncols threads = nthreads
    end
    if cuda_devices > 0 && isa(pu, CPU)
        @warn "CUDA is available but input is on CPU; GPU not used."
    end
    if cuda_devices > 1 && isa(pu, GPU)
        @warn "Multiple CUDA devices available; using default device only." devices = cuda_devices
    end
    return nothing
end
