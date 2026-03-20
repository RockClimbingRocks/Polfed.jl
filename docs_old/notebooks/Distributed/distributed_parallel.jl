using Distributed

# ==========================================================
# 1. SETUP THE HYBRID ENVIRONMENT (This part was correct)
# ==========================================================
println("--- Setting up Environment ---")
println("Main process #$(myid()) started with $(Threads.nthreads()) threads.")

println("Adding 4 worker processes, each with 2 threads...")
addprocs(4; exeflags="--threads=2")
println("Total processes now: $(nprocs()). Worker IDs: $(workers())")

# ==========================================================
# 2. DEFINE THE WORK TO BE DONE (This part was correct)
# ==========================================================
@everywhere using Base.Threads

@everywhere function perform_hybrid_task(task_id)
    pid = myid()
    num_threads = nthreads()
    
    println("  -> Task $task_id running on Process #$pid, which has $num_threads threads available.")
    
    local_sum = Threads.Atomic{Float64}(0.0)
    @threads for i in 1:2
        tid = threadid() 
        println("    -> Thread $tid is processing iteration $i of Task $task_id")
        Threads.atomic_add!(local_sum, sin(i * task_id))
    end

    println("------- Task $task_id complete -------")

    return "Result from Task $task_id (processed by PID $pid)"
end

# ==========================================================
# 3. DISTRIBUTE THE WORK (This is the corrected section)
# ==========================================================
println("\n--- Distributing Work ---")

# 1. Create the list of process IDs that INCLUDES the main process
pids_with_main = workers()
println("Creating a worker pool with PIDs: $(pids_with_main)")

# 2. Create an explicit WorkerPool object from this list
pool = WorkerPool(pids_with_main)

# Dummy data: we have 5 tasks to run on our 5 available processes
tasks_to_run = 1:6

# 3. Pass the custom pool as the SECOND ARGUMENT to pmap
# The syntax is: pmap(function, pool, data)
results = pmap(perform_hybrid_task, pool, tasks_to_run)

# ==========================================================
# 4. SHOW RESULTS (This part was correct)
# ==========================================================
println("\n--- All tasks complete ---")
println("Results:")
for res in results
    println(res)
end