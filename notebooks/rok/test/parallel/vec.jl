

using LinearAlgebra, MKL_jll, MKLSparse, BenchmarkTools, Polyester, LoopVectorization
include("../qrem.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

@inline function vec!(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    @inbounds @. b1 = c * X + α * b1 + β * b3
end


@inline function vec!_poly(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    @batch for i in eachindex(b1)
        b1[i] = c * X[i] + α * b1[i] + β * b3[i]
    end
end



@inline function vec!_poly_inbounds(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    @inbounds @batch for i in eachindex(b1)
        b1[i] = c * X[i] + α * b1[i] + β * b3[i]
    end
end


@inline function vec!_vec(
    mapping!::Function,
    b1::AbstractVecOrMat{<:Number},
    b2::AbstractVecOrMat{<:Number},
    b3::AbstractVecOrMat{<:Number},
    c::Number,
    α::Real,
    β::Real,
    X::AbstractVecOrMat{<:Number},
)
    @tturbo @. b1 = c * X + α * b1 + β * b3
end


@inline function _mkl_combine!(
    b1::Array{Float64},
    X::Array{Float64},
    b3::Array{Float64},
    c::Number,
    α::Real,
    β::Real,
)
    length(b1) == length(X) == length(b3) ||
        throw(DimensionMismatch("Clenshaw buffers must have equal lengths"))

    n   = Cint(length(b1))
    inc = Cint(1)
    cₘ  = Cdouble(c)
    αₘ  = Cdouble(α)
    βₘ  = Cdouble(β)

    # b1 = c*X + α*b1
    @ccall libmkl_rt.cblas_daxpby(
        n::Cint,
        cₘ::Cdouble,
        X::Ptr{Cdouble},
        inc::Cint,
        αₘ::Cdouble,
        b1::Ptr{Cdouble},
        inc::Cint,
    )::Cvoid

    # b1 = β*b3 + b1
    @ccall libmkl_rt.cblas_daxpy(
        n::Cint,
        βₘ::Cdouble,
        b3::Ptr{Cdouble},
        inc::Cint,
        b1::Ptr{Cdouble},
        inc::Cint,
    )::Cvoid

    return b1
end


@inline function vec!_mkl(
    mapping!::Function,
    b1::Array{Float64},
    b2::Array{Float64},
    b3::Array{Float64},
    c::Number,
    α::Real,
    β::Real,
    X::Array{Float64},
)

    _mkl_combine!(b1, X, b3, c, α, β)
end






L=16
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

# print("ROW mapping: ")
# @btime mapping_row(b1_row, X_row)  
# print("ROW vec: ")
# @btime vec!(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
# print("ROW combined: ")
# @btime mapping_recurrence!(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)

# print("COL mapping: ")
# @btime mapping_col(b1_col, X_col)
# print("COL vec: ")
# @btime vec!(mapping_col, b1_col, b2_col, b3_col, 1.0, 2.0, -1.0, X_col)
# print("COL combined: ")
# @btime mapping_recurrence!(mapping_col, b1_col, b2_col, b3_col, 1.0, 2.0, -1.0, X_col)




println("Different vecotirzations:")
print("ROW vec (base): ")
@btime vec!(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
print("ROW vec (poly): ")
@btime vec!_poly(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
print("VOL vec (poly): ")
@btime vec!_poly(mapping_col, b1_col, b2_col, b3_col, 1.0, 2.0, -1.0, X_col)
print("ROW vec (poly-@inbounds): ")
@btime vec!_poly_inbounds(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
print("ROW vec (vec): ")
@btime vec!_vec(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
print("ROW vec (mkl): ")
@btime vec!_mkl(mapping_row, b1_row, b2_row, b3_row, 1.0, 2.0, -1.0, X_row)
