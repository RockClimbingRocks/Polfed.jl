using LinearAlgebra
using Polyester
using BenchmarkTools
using Base.Threads

# (The `do_work` and other transform functions remain the same...)
function do_work(y_out, y_in, b_vec)
    for i in eachindex(y_out, y_in)
        y_out[i] = sin(y_in[i] * b_vec[1]) + log(abs(cos(y_in[i] - b_vec[2])))
    end
end
function transform_serial(Ỹi, Yi, bi); do_work(Ỹi, Yi, bi); end
function transform_polyester(Ỹi, Yi, bi); @batch for i in eachindex(Ỹi, Yi); Ỹi[i] = sin(Yi[i] * bi[1]) + log(abs(cos(Yi[i] - bi[2]))); end; end


#------------ D. Generalized Manual `Threads.@spawn` `transform` ------------

function transform_spawn_n(Ỹi, Yi, bi, n_chunks)
    len = length(Ỹi)
    if len == 0 || n_chunks <= 1
        do_work(Ỹi, Yi, bi)
        return
    end

    @sync begin
        for i in 1:n_chunks
            chunk_size = cld(len, n_chunks) 
            start_idx = (i - 1) * chunk_size + 1
            end_idx = min(i * chunk_size, len)
            
            if start_idx > len
                break
            end

            Threads.@spawn begin
                y_out_view = view(Ỹi, start_idx:end_idx)
                y_in_view = view(Yi, start_idx:end_idx)
                do_work(y_out_view, y_in_view, bi)
            end
        end
    end
end

# (The `clenshaw` function remains exactly the same...)
function clenshaw(transform_func, Ỹ, Y, b)
    @assert size(Y) == size(Ỹ)
    @assert length(b) == size(Y, 2)
    ncols = size(Y, 2)
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @threads for i in 1:ncols
        transform_func(view(Ỹ, :, i), view(Y, :, i), b[i])
    end
    BLAS.set_num_threads(prev_blas_threads)
    return Ỹ
end


# ===================================================================
# 3. Main Execution and Benchmarking
# ===================================================================
function main()
    println("Using $(Threads.nthreads()) total threads.")

    # Setup sample data
    nrows = 1_000_000
    ncols = 2
    Y = rand(nrows, ncols)
    b = [rand(2) for _ in 1:ncols]

    # --- Run and Benchmark ---

    # 1. Fully Serial Version
    println("\nRunning fully serial version...")
    Ỹ_serial = similar(Y)
    @btime for i in 1:$ncols; $transform_serial(view($Ỹ_serial, :, i), view($Y, :, i), $b[i]); end

    # 2. Nested Parallelism with Polyester
    println("\nRunning nested parallelism with Polyester.jl...")
    Ỹ_polyester = similar(Y)
    @btime $clenshaw($transform_polyester, $Ỹ_polyester, $Y, $b)
    
    # 3. Generalized Spawn Version
    n_threads_per_column = 3
    total_tasks = ncols * n_threads_per_column
    println("\nRunning generalized spawn with $n_threads_per_column threads/column (Total tasks: $total_tasks)...")
    
    # Create the wrapper function (closure)
    transform_wrapper = (Ỹi, Yi, bi) -> transform_spawn_n(Ỹi, Yi, bi, n_threads_per_column)
    
    Ỹ_spawn_n = similar(Y)
    @btime $clenshaw($transform_wrapper, $Ỹ_spawn_n, $Y, $b)

    # --- Verification ---
    println("\nVerifying results...")
    @assert Ỹ_polyester ≈ Ỹ_spawn_n "Polyester and Spawn(n) results do not match!"
    serial_result = similar(Y)
    for i in 1:ncols; transform_serial(view(serial_result, :, i), view(Y, :, i), b[i]); end
    @assert Ỹ_polyester ≈ serial_result "Parallel result does not match serial result!"
    println("All methods produced identical results. Benchmarking complete.")
end

main()