# distributed_mini_example.jl

module MiniPolfed

using Distributed
using LinearAlgebra, SparseArrays, SharedArrays
using Base.Threads

# These are at the TOP LEVEL. This is correct.
include("../../../src/Clenshaw/ClenshawMapping.jl")
using .ClenshawMapping

# __init__ is removed. We will now handle this explicitly.

"""
    set_workers(requested_workers::Int)

The definitive setup function. It adjusts the worker pool to the desired size
and ensures all workers have the MiniPolfed module fully loaded by re-including
this entire file on them.
"""
function set_workers(requested_workers::Int)
    if requested_workers < 1
        error("Number of workers must be at least 1.")
    end

    current_workers = nworkers()

    # Step 1: Add or remove workers to match the request.
    if current_workers < requested_workers
        num_to_add = requested_workers - current_workers
        println("Info: Adding $(num_to_add) worker(s) to reach the target of $(requested_workers).")
        addprocs(num_to_add)
    elseif current_workers > requested_workers
        num_to_remove = current_workers - requested_workers
        procs_to_remove = workers()[end - num_to_remove + 1:end]
        println("Info: Removing $(num_to_remove) worker(s) to reach the target of $(requested_workers).")
        rmprocs(procs_to_remove)
    end

    # Step 2: This is the key. We use the same working pattern as your __init__.
    # It loads this entire file on all workers, which correctly processes the
    # top-level `using` statements. This avoids the "toplevel" error.
    println("Info: Loading module code on all $(nworkers()) active workers.")
    @everywhere workers() include($(@__FILE__))
end

function perform_hybrid_task(Y::AbstractVecOrMat, X::AbstractVecOrMat, b::Vector{<:AbstractVecOrMat{<:Real}}, f!::Function)
    pid = myid()
    num_threads = nthreads()
    
    println("  -> Task running on Process #$pid, which has $num_threads threads available.")
    # On the worker, f! will call mul!(Y, mat, X). `mat` needs to exist there.
    # The clenshaw function itself is serialized and sent to the worker.
    f!(Y, X, b)
    println("------- Task completed on PID #$pid -------")
    return "Result from Task (processed by PID $pid)"
end


function run_example(ncols; nworkers::Int = 4)
    set_workers(nworkers)
    
    println("Running example with $(ncols) tasks on a pool of $(nworkers) workers.")

    D = 2^16

    # --- Step 1: Create ONE definitive matrix on the main process ---
    println("Creating the definitive matrix on the main process...")
    mat_local = sprandn(D, D, 0.01)
    mat_local = mat_local + mat_local' # Make it symmetric

    # --- Step 2: Distribute this exact matrix to all workers ---
    println("Distributing the matrix to all workers...")
    @everywhere workers() mat_for_worker = $mat_local

    # --- Step 3: Define ONE Clenshaw transformation ---
    # This is the key. We define a single, consistent transformation.
    T = Float64
    order = 10
    coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n,0)) * cos(n * acos(λ)))
    bs = [[zeros(T, D) for _ in 1:3] for _ in 1:ncols]

    # This `f!` will be used by the serial version. It captures `mat_local`.
    f!_on_main = (Y, X) -> mul!(Y, mat_local, X)
    clenshaw_transform = (Y,X,b) -> ClenshawMapping.Clenshaw(:Chebyshev, n->coefficients(0.,n), order, f!_on_main, D, T)(Y,X,b)

    # This is a generic task definition for the parallel version.
    # It tells the worker to use its own `mat_for_worker`.
    f!_on_worker = (Y, X) -> mul!(Y, Main.mat_for_worker, X)
    parallel_task = (Y,X,b) -> ClenshawMapping.Clenshaw(:Chebyshev, n->coefficients(0.,n), order, f!_on_worker, D, T)(Y,X,b)

    X = Matrix(qr(randn(D, ncols)).Q)

    # Use SharedArrays for input (X) and output (Y1) for the parallel run
    X_sm = SharedArray(X)
    Y1 = SharedArray(X_sm)
    Y2 = similar(X)

    # --- Step 4: Execute the transformations ---

    println("\n--- Running pmap (parallel) ---")
    # We pass the generic `parallel_task` to pmap.
    @time pmap(i -> perform_hybrid_task(view(Y1,:,i), view(X_sm,:,i), bs[i], parallel_task), 1:ncols)

    println("\n--- Running map (serial on main process) ---")
    # We use the `clenshaw_transform` defined with the local matrix.
    @time map(i -> clenshaw_transform(view(Y2,:,i), view(X,:,i), bs[i]), 1:ncols)

    display(Y1[1:10,:])
    display(Y2[1:10,:])
    println("\nAre they the same? : ", Y1 ≈ Y2)
end
end # module