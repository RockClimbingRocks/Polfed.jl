using LinearAlgebra
using Polyester
using BenchmarkTools
using Base.Threads

# Dummy type to match your function signature
struct ClenshawType end

# ===================================================================
# 1. The Inner `transform` Functions (Serial, Polyester, Spawn)
# ===================================================================

# This is a placeholder for your actual column transformation.
# We add some CPU-intensive work to make the parallelism meaningful.
function do_work(y_out, y_in, b_vec)
    for i in eachindex(y_out, y_in)
        # Some arbitrary, heavy computation
        y_out[i] = sin(y_in[i] * b_vec[1]) + log(abs(cos(y_in[i] - b_vec[2])))
    end
end

# ------------ A. Serial Transform (for baseline) ------------
function transform_serial(Ỹi, Yi, bi)
    # This version is not parallel at all.
    do_work(Ỹi, Yi, bi)
end

# ------------ B. Polyester-based `transform` for nested parallelism ------------
# This is the recommended approach.
function transform_polyester(Ỹi, Yi, bi)
    # @batch is safe to nest inside an outer @threads loop.
    # It will efficiently schedule the work on the available threads
    # without oversubscribing.
    @batch for i in eachindex(Ỹi, Yi)
        # Some arbitrary, heavy computation
        Ỹi[i] = sin(Yi[i] * bi[1]) + log(abs(cos(Yi[i] - bi[2])))
    end
end

# ------------ C. Manual `Threads.@spawn` `transform` ------------
# This version gives you explicit control to split the work in two.
function transform_spawn(Ỹi, Yi, bi)
    len = length(Ỹi)
    mid = len ÷ 2

    # @sync waits for all spawned tasks within its block to complete.
    @sync begin
        # Spawn a task for the first half of the column
        Threads.@spawn begin
            y_out_view = view(Ỹi, 1:mid)
            y_in_view = view(Yi, 1:mid)
            do_work(y_out_view, y_in_view, bi)
        end

        # The main thread works on the second half concurrently
        y_out_view = view(Ỹi, mid+1:len)
        y_in_view = view(Yi, mid+1:len)
        do_work(y_out_view, y_in_view, bi)
    end
end


# ===================================================================
# 2. The Outer `clenshaw` Functions
# ===================================================================

# This is your outer-loop parallel function, now generalized.
# It takes the specific `transform_func` as an argument.
function clenshaw(
    transform_func, # Pass the transform function to use
    Ỹ::AbstractVecOrMat,
    Y::AbstractVecOrMat,
    b::AbstractVector,
)
    @assert size(Y) == size(Ỹ) "Y and Ỹ must be of the same size!"
    @assert length(b) == size(Y, 2) "Length of b must be the same as the number of columns of Y!"

    ncols = size(Y, 2)
    
    # Good practice: prevent BLAS from oversubscribing threads when you
    # are already managing parallelism at a higher level.
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    # The outer loop parallelizes over columns.
    @threads for i in 1:ncols
        Ỹi = view(Ỹ, :, i)
        Yi = view(Y, :, i)
        # IMPORTANT FIX: You should use b[i] for the i-th column, not b[t].
        transform_func(Ỹi, Yi, b[i])
    end
    
    BLAS.set_num_threads(prev_blas_threads)
    return Ỹ
end


# ===================================================================
# 3. Main Execution and Benchmarking
# ===================================================================
function main()
    println("Using $(Threads.nthreads()) threads.")

    # Setup sample data
    nrows = 50_000
    ncols = 2
    Y = rand(nrows, ncols)
    Ỹ = similar(Y)
    
    # Create a vector of vectors for `b`
    # Each column gets its own 2-element parameter vector
    b = [rand(2) for _ in 1:ncols]

    # --- Run and Benchmark ---

    # 1. Fully Serial Version (for baseline)
    println("\nRunning fully serial version...")
    Ỹ_serial = similar(Y)
    @btime for i in 1:$ncols
        $transform_serial(view($Ỹ_serial, :, i), view($Y, :, i), $b[i])
    end

    # 2. Nested Parallelism with Polyester
    println("\nRunning nested parallelism with Polyester.jl...")
    Ỹ_polyester = similar(Y)
    @btime $clenshaw($transform_polyester, $Ỹ_polyester, $Y, $b)

    # 3. Nested Parallelism with manual @spawn
    println("\nRunning nested parallelism with Threads.@spawn...")
    Ỹ_spawn = similar(Y)
    @btime $clenshaw($transform_spawn, $Ỹ_spawn, $Y, $b)
    
    # --- Verification ---
    @assert Ỹ_polyester ≈ Ỹ_spawn "Polyester and Spawn results do not match!"
    # Compare against the serial result to ensure correctness
    serial_result = similar(Y)
    for i in 1:ncols
        transform_serial(view(serial_result, :, i), view(Y, :, i), b[i])
    end
    @assert Ỹ_polyester ≈ serial_result "Parallel result does not match serial result!"

    println("\nAll methods produced identical results. Benchmarking complete.")
end

main()