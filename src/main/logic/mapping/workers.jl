# Distributed worker lifecycle for the two-level parallelization strategy. Split from
# parallel.jl because it is the only part of the mapping slice that touches the
# `Distributed` stdlib (worker spawn/teardown); everything else here is thread-local.

"""
    set_workers(x0, strategy) -> nothing

Initialize Distributed workers for `strategy`.

No-op for [`NoParallel`](@ref), [`MulColsParallel`](@ref), [`MulRowsParallel`](@ref),
and `nothing`. For [`TwoLevelParallel`](@ref), spawns one worker process per block
column (`1` for a vector), each launched with the active project and `nt_per_col`
threads, with POLFED loaded (`using UniPolfed`) so the workers can run the transform.
"""
set_workers(::AbstractVecOrMat, ::Union{NoParallel,MulColsParallel,MulRowsParallel}) = nothing
set_workers(::AbstractVecOrMat, ::Nothing) = nothing

function set_workers(x0::AbstractVecOrMat, strategy::TwoLevelParallel)
    requested_workers = x0 isa AbstractVector ? 1 : size(x0, 2)
    threads_per_worker = strategy.nt_per_col
    project_path = Base.active_project()

    @info "Initializing workers for two-level parallelization." requested_workers threads_per_worker project_path
    w = addprocs(requested_workers; exeflags = ["--project=$(project_path)", "--threads=$(threads_per_worker)"])
    # load POLFED on the new workers; `remotecall_eval` takes `using` as quoted data,
    # so (unlike `@everywhere w using UniPolfed`) it is valid inside a function body.
    remotecall_eval(Main, w, :(using UniPolfed))

    strategy.worker_pool = WorkerPool(w)
    strategy.clenshaw_shared_buffer = nothing
    return nothing
end

"""
    remove_workers(strategy) -> nothing

Tear down workers for `strategy`.

No-op for [`NoParallel`](@ref), [`MulColsParallel`](@ref), [`MulRowsParallel`](@ref),
and `nothing`. For [`TwoLevelParallel`](@ref), removes the pooled worker processes
and resets the pool and shared buffer.
"""
remove_workers(::Union{NoParallel,MulColsParallel,MulRowsParallel}) = nothing
remove_workers(::Nothing) = nothing

function remove_workers(strategy::TwoLevelParallel)
    rmprocs(collect(strategy.worker_pool.workers))
    strategy.worker_pool = WorkerPool()
    strategy.clenshaw_shared_buffer = nothing
    return nothing
end
