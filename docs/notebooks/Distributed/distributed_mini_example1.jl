# distributed_mini_example.jl

module MiniPolfed

using Distributed


include("../../../src/ClenshawMapping/ClenshawMapping.jl")

using .ClenshawMapping
using LinearAlgebra, SparseArrays
using Base.Threads

function __init__()
    # This function runs when the module is loaded on ANY process.
    # We MUST guard the setup logic to prevent recursion on workers.
    if myid() == 1
        # Only the main process (ID = 1) should ever set up workers.

        # Step 1: Add worker processes if we are in a single-process environment.
        if nprocs() == 1
            println("Info: MiniPoslfed is automatically adding 4 worker processes.")
            addprocs(4)
        end

        # Step 2: Load this module's code onto all newly added worker processes.
        # This will cause __init__() to run on the workers, but the myid()==1
        # check will prevent them from doing anything.
        @everywhere workers() include($(@__FILE__))
    end
end


function perform_hybrid_task(Y::AbstractVecOrMat, X::AbstractVecOrMat, b::Vector{<:AbstractVecOrMat{<:Real}}, f!::Function)
    pid = myid()
    num_threads = nthreads()
    
    println("  -> Task running on Process #$pid, which has $num_threads threads available.")
    f!(Y, X, b)
    println("------- Task completed -------")
    return "Result from Task (processed by PID $pid)"
end


function run_example()
    pids = workers()
    ncols = length(pids) # Assuming one column per worker
    println("Creating a worker pool with PIDs: $(pids)")

    D = 2^14
    mat1 = sprandn(D,D,0.01)
    mat = (mat1 +mat1' .- 0) ./100
    # @everywhere mat = randn(D,D)

    f! = (Y, X) -> mul!(Y, mat, X)  # dummy mapping function (matrix multiplication)

    T = eltype(mat)
    order = 10
    coefficients(λ::Real, n::Int) = (2 - ==(n,0)) * cos(n * acos(λ))
    bs = [[zeros(T, D) for _ in 1:3] for _ in 1:ncols]
    clenshaw = (Y,X,b) -> ClenshawMapping.Clenshaw(:Chebyshev, n->coefficients(0.,n), order, f!, D, T)(Y,X,b)



    X = Matrix(qr(randn(D, ncols)).Q)
    Y1 = similar(X)
    Y2 = similar(X)
    pmap(i -> perform_hybrid_task(view(Y1,:,i), view(X,:,i), bs[i], clenshaw), 1:ncols)
    @time pmap(i -> perform_hybrid_task(view(Y1,:,i), view(X,:,i), bs[i], clenshaw), 1:ncols)


    # @btime $clenshaw(view($Y,:,1), view($X,:,1), $bs[1])
    map(i -> clenshaw(view(Y2,:,i), view(X,:,i), bs[i]), 1:ncols)
    @time map(i -> clenshaw(view(Y2,:,i), view(X,:,i), bs[i]), 1:ncols)

    display(Y1[1:10,:])
    display(Y2)
    println("Are they the same? : ", Y1 ≈ Y2)
    println(maximum(abs.(Y1 .- Y2)))
end



end # module