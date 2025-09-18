using Distributed
using LinearAlgebra
using SharedArrays
include("../../../src/ClenshawMapping/ClenshawMapping.jl")
using .ClenshawMapping

ncols = 4
threads_per_worker = 2
addprocs(ncols; exeflags="--threads=$(threads_per_worker)")

# --- Make module visible on all workers ---
@everywhere begin
    include("../../../src/ClenshawMapping/ClenshawMapping.jl")
    using .ClenshawMapping
    using SharedArrays
    using LinearAlgebra
end


# ----- CONFIG -----
ncols = 4
threads_per_worker = 2
repetitions = 1000   # how many times each worker will apply the kernel
n = 200              # problem size (rows)

# spawn workers (each with threads_per_worker threads)
addprocs(ncols; exeflags="--threads=$(threads_per_worker)")


# ----- DRIVER (runs on master) -----
function main()
    # Build sample input
    A = randn(n, n)           # (not used by the dummy clenshaw_algorithm! here)
    X = randn(n, ncols)

    mapping_per_col! = (Y,X) -> mul!(Y, A, X)  # dummy mapping function (matrix multiplication)
    coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n,0)) * cos(n * acos(λ)))
    order = 1000
    D = size(A,1)
    T = eltype(A)

    bcols = [ [zeros(Float64, n) for _ in 1:3] for _ in 1:ncols]
    clenshaw = ClenshawMapping.Clenshaw(:Chebyshev, n->coefficients(0.,n), order, mapping_per_col!, D, T)



    Y = SharedArray{Float64}((D, ncols))

    pmap(j -> clenshaw(view(Y,:,j), view(X,:,j), b[j]), 1:ncols)


end


main()