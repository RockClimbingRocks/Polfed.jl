
abstract type Parallelization end


mutable struct NoParallel<:Parallelization end

mutable struct MulColsParallel<:Parallelization end

mutable struct TwoLevelParallel<:Parallelization 
    worker_pool::WorkerPool  # Vector of worker process IDs
    nt_per_col::Int 

    function TwoLevelParallel(nt_per_col::Int)
        new(WorkerPool(), nt_per_col)
    end
end