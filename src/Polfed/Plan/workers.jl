
"""
    set_workers(x0, parallel_strategy) -> nothing

Initialize distributed workers for the selected parallelization strategy.

For [`NoParallel`](@ref), [`MulColsParallel`](@ref), and `Nothing`, this is a
no-op. For [`TwoLevelParallel`](@ref), workers are spawned and initialized.
"""
function set_workers(_::AbstractVecOrMat, _::Union{NoParallel,MulColsParallel})
    return nothing
end

function set_workers(_::AbstractVecOrMat, ::Nothing)
    return nothing
end

"""
    set_workers(x0::AbstractVecOrMat, parallel_strategy::TwoLevelParallel) -> nothing

Spawn one worker per input column (`1` for vector input), configure worker
thread count from `parallel_strategy.nt_per_col`, and include required POLFED
files on each worker.
"""
function set_workers(x0::AbstractVecOrMat, parallel_strategy::TwoLevelParallel)
    requested_workers = x0 isa AbstractVector ? 1 : size(x0,2)
    threads_per_worker = parallel_strategy.nt_per_col


    project_path = Base.active_project()
    @info "Initializing workers for two-level parallelization." requested_workers threads_per_worker project_path
    @debug "Workers before addprocs." workers=workers()
    w = addprocs(requested_workers; exeflags=["--project=$(project_path)" ,"--threads=$threads_per_worker"])
    @debug "Workers after addprocs." workers=workers() new_workers=w
    @debug "Main module file." main_module_file

    @debug "Including main module file on workers."
    @everywhere w include($main_module_file)

    parallel_strategy.worker_pool = WorkerPool(w)
    parallel_strategy.clenshaw_shared_buffer = nothing
end




"""
    remove_workers(parallel_strategy) -> nothing

Tear down workers for the selected strategy.

For [`NoParallel`](@ref), [`MulColsParallel`](@ref), and `Nothing`, this is a
no-op.
"""
function remove_workers(_::Union{NoParallel,MulColsParallel})
    return nothing
end

function remove_workers(::Nothing)
    return nothing
end

"""
    remove_workers(parallel_strategy::TwoLevelParallel) -> nothing

Remove all workers held in `parallel_strategy.worker_pool` and reset shared
worker-side state.
"""
function remove_workers(parallel_strategy::TwoLevelParallel)

    # println("parallel_strategy: ", parallel_strategy)
    # println("worker_pool: ", parallel_strategy.worker_pool)
    # println("Removing workers: ", parallel_strategy.worker_pool.workers)
    workers = collect(parallel_strategy.worker_pool.workers)
    rmprocs(workers)
    parallel_strategy.worker_pool = WorkerPool()
    parallel_strategy.clenshaw_shared_buffer = nothing
end
