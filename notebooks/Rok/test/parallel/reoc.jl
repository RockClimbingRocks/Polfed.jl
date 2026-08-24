

using LinearAlgebra, MKL_jll, MKLSparse, BenchmarkTools, Polyester, LoopVectorization
include("../qrem.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0



@inline function mapping_recurrence!_base(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    mapping!(b1, b2)
    @inbounds @. b1 = c * X + α * b1 + β * b3
    return b1
end


@inline function mapping_recurrence!_poly(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    mapping!(b1, b2)
    @batch for i in eachindex(b1)
        b1[i] = c * X[i] + α * b1[i] + β * b3[i]
    end
    return b1
end









L=18
bs = 4


mat = qrem_hamiltonian(L; Γ=1.0, rng=MersenneTwister(42))
# display(mat)


mapping_row = (Y, X) -> mul!(Y, X, mat)
mapping_col = (Y, X) -> mul!(Y, mat, X)

X_row  = rand(bs, 2^L)
b1_row = rand(bs, 2^L)
b2_row = rand(bs, 2^L)
b3_row = rand(bs, 2^L)

X_col  = rand(2^L, bs)
b1_col = rand(2^L, bs)
b2_col = rand(2^L, bs)
b3_col = rand(2^L, bs)




println("threads = $(Threads.nthreads()), mkl threads = $(get(ENV, "MKL_NUM_THREADS", "not set")) ")

print("ROW base: ")
@btime mapping_recurrence!_base(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
print("ROW poly: ")
@btime mapping_recurrence!_poly(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)



