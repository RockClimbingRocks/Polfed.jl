
function set_workers(_::AbstractVecOrMat, _::Union{NoParallel,MulColsParallel})
    return nothing
end

function set_workers(x0::AbstractVecOrMat, parallel_strategy::TwoLevelParallel)
    requested_workers = x0 isa AbstractVector ? 1 : size(x0,2)
    threads_per_worker = parallel_strategy.nt_per_col


    project_path = Base.active_project()
    println("Project path: ", project_path)
    println("Workers: ", workers())
    w = addprocs(requested_workers; exeflags=["--project=$(project_path)" ,"--threads=$threads_per_worker"])
    println("Workers: ", workers())
    println("New workers: ", w)


    println("seronja")
    println("Main module file: ", main_module_file)

    println("Including files on workers...")
    current_folder = dirname(@__FILE__)
    workers_using_path = joinpath(current_folder, "workers_using.jl")
    println("workers_using.jl path: ", workers_using_path)
    @everywhere w include($workers_using_path)
    println("Included workers_using.jl")
    @everywhere w include($main_module_file)

    parallel_strategy.worker_pool = WorkerPool(w)
end




function remove_workers(_::Union{NoParallel,MulColsParallel})
    return nothing
end

function remove_workers(parallel_strategy::TwoLevelParallel)

    # println("parallel_strategy: ", parallel_strategy)
    # println("worker_pool: ", parallel_strategy.worker_pool)
    # println("Removing workers: ", parallel_strategy.worker_pool.workers)
    workers = collect(parallel_strategy.worker_pool.workers)
    rmprocs(workers)
end