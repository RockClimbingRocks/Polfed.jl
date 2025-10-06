
function set_workers(_::AbstractVecOrMat, _::Union{NoParallel,MulColsParallel})
    return nothing
end

using Distributed

function set_workers(x0::AbstractVecOrMat, parallel_strategy::TwoLevelParallel)
    requested_workers = x0 isa AbstractVector ? 1 : size(x0,2)
    threads_per_worker = parallel_strategy.nt_per_col

    # --- Use the project directory, not the Project.toml file ---
    project_file = Base.active_project()
    project_dir = isdir(project_file) ? project_file : dirname(project_file)
    println("Project dir: ", project_dir)

    println("Workers before: ", workers())
    w = addprocs(requested_workers;
                 exeflags=["--project=$(project_dir)", "--threads=$threads_per_worker"])
    println("Workers after: ", workers())
    println("New workers: ", w)

    # --- Activate the project on all workers ---
    @everywhere begin
        import Pkg
        Pkg.activate($project_dir)
        # Optional: ensure packages are installed on workers
        # Pkg.instantiate()
    end

    println("Activated project on workers...")

    # --- Include your module file on the workers ---
    @everywhere include($main_module_file)

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