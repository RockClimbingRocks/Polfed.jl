
"""
Abstract base type for all parallelization strategies.

All parallelization types should inherit from `Parallelization`.
It is used to define interfaces and enable multiple dispatch
based on the chosen strategy.
"""
abstract type Parallelization end

"""
Represents a serial execution strategy with **no parallelization**.

Useful for debugging, testing, when parallelism is not required, and mostly when all of the parallelization is handled by the user (for example, when using GPUs).
"""
mutable struct NoParallel <: Parallelization end

"""
Represents **multi-column parallelization**, where independent columns
of a matrix or data are processed in parallel.

Each column can be handled by a separate worker or thread,
providing speedup when the number of columns is large.
This strategy assumes that column operations are independent.
"""
mutable struct MulColsParallel <: Parallelization
    nt_per_col::Int

    function MulColsParallel(nt_per_col::Int=1)
        new(nt_per_col)
    end
end

"""
Represents a **two-level parallelization** strategy.

1. **Inter-column parallelism:** Columns are distributed among a pool of worker processes.
2. **Intra-column parallelism:** Within each column, multiple threads (or tasks) perform computations in parallel.

# Fields
- `worker_pool::WorkerPool`: A collection of worker processes handling column-level tasks.
- `nt_per_col::Int`: Number of threads per column for intra-column parallelism.

# Constructor
```julia
TwoLevelParallel(nt_per_col::Int)
```

Creates a TwoLevelParallel instance with a new worker pool and
the specified number of threads per column.

This strategy is ideal for large-scale computations where both
the number of columns and the per-column workload are significant.
"""
mutable struct TwoLevelParallel <: Parallelization
    worker_pool::WorkerPool  # Vector of worker process IDs
    nt_per_col::Int 
    clenshaw_shared_buffer::Any

    function TwoLevelParallel(nt_per_col::Int)
        new(WorkerPool(), nt_per_col, nothing)
    end
end
