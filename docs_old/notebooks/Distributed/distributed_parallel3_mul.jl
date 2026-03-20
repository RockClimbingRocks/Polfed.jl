using Distributed, LinearAlgebra, Base.Threads

# Ensure all workers load required modules
@everywhere using LinearAlgebra, Base.Threads

# Function: multiply matrix with vector 1000 times, using all threads in this process
@everywhere function map_vector!(A::Matrix{Float64}, x::Vector{Float64})
    y = similar(x)
    for rep in 1:1000
        # parallel matvec using all available threads in this process
        n = size(A,1)
        @sync for t in 1:nthreads()
            Threads.@spawn begin
                # row chunk for this thread
                rows_per_thread = cld(n, nthreads())
                start_row = (t-1)*rows_per_thread + 1
                end_row   = min(t*rows_per_thread, n)
                @views mul!(y[start_row:end_row], A[start_row:end_row, :], x)
            end
        end
        # swap x and y for next iteration
        x, y = y, x
    end
    return x
end

# Driver
function main()
    # small matrix
    n = 100
    A = randn(n,n)

    # 4 different vectors
    xs = [randn(n) for _ in 1:4]

    # Distribute work: each vector -> one process
    results = pmap(xs) do x
        map_vector!(A, x)
    end

    println("Computation finished. Norms of results: ", [norm(r) for r in results])
end

main()