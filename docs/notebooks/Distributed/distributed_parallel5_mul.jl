using Distributed
using LinearAlgebra
using SparseArrays
using SharedArrays
using BenchmarkTools
nt = 2

println("Main process #$(myid()) with $(Threads.nthreads()) threads")
addprocs(4; exeflags="--threads=$(nt)")
println("Workers: ", workers())

# --- Include local module on all processes ---
@everywhere include("../../../src/ClenshawMapping/ClenshawMapping.jl")
@everywhere using .ClenshawMapping

# --- Load other modules everywhere ---
@everywhere using Base.Threads, LinearAlgebra, SharedArrays, SparseArrays


@everywhere function perform_hybrid_task(Y::AbstractVecOrMat, X::AbstractVecOrMat, b::Vector{<:AbstractVecOrMat{<:Real}}, f!::Function)
    pid = myid()
    num_threads = nthreads()
    
    println("  -> Task running on Process #$pid, which has $num_threads threads available.")
    f!(Y, X, b)
    println("------- Task completed -------")
    return "Result from Task (processed by PID $pid)"
end


# ==========================================================
# 3. DISTRIBUTE THE WORK (This is the corrected section)
# ==========================================================
println("\n--- Distributing Work ---")

# 1. Create the list of process IDs that INCLUDES the main process
@everywhere pids = workers()
@everywhere ncols = length(pids) # Assuming one column per worker
println("Creating a worker pool with PIDs: $(pids)")

@everywhere D = 2^12
@everywhere mat = sprandn(D,D,0.01)
# @everywhere mat = randn(D,D)
@everywhere f! = (Y, X) -> begin
    nt = 2

    @threads for t in 1:nt
        chunk_size = cld(size(Y,1), nt) 
        start_row = (t - 1) * chunk_size + 1
        end_row = min(t * chunk_size, size(Y,1))
        if start_row > size(Y,1)
            break
        end

        Y_t = view(Y, start_row:end_row, :)
        mat_t = view(mat, start_row:end_row, :)
        mul!(Y_t, mat_t, X)  # dummy mapping function (matrix multiplication)
    end
end
@everywhere T = eltype(mat)
@everywhere order = 1000
@everywhere coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n,0)) * cos(n * acos(λ)))
@everywhere bs = [[zeros(T, D) for _ in 1:3] for _ in 1:ncols]
@everywhere clenshaw = (Y,X,b) -> ClenshawMapping.Clenshaw(:Chebyshev, n->coefficients(0.,n), order, f!, D, T)(Y,X,b)


X = randn(D, ncols)
Y = similar(X)
pmap(i -> perform_hybrid_task(view(Y,:,i), view(X,:,i), bs[i], clenshaw), 1:ncols)
@time pmap(i -> perform_hybrid_task(view(Y,:,i), view(X,:,i), bs[i], clenshaw), 1:ncols)

# X = SharedArray{Float64,2}(X)
# Y = similar(X)
# pmap(i -> perform_hybrid_task(view(Y,:,i), view(X,:,i), clenshaw), 1:ncols)
# @time pmap(i -> perform_hybrid_task(view(Y,:,i), view(X,:,i), clenshaw), 1:ncols)


X = randn(D, ncols)
Y = similar(X)

clenshaw = (Y,X,b) -> ClenshawMapping.Clenshaw(:Chebyshev, n->coefficients(0.,n), order, f!, D, T)(Y,X,b)
# @btime $clenshaw(view($Y,:,1), view($X,:,1), $bs[1])
clenshaw(view(Y,:,1), view(X,:,1), bs[1])
@time clenshaw(view(Y,:,1), view(X,:,1), bs[1])

# 110sec