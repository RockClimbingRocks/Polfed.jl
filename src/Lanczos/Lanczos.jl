module Lanczos

using LinearAlgebra, CUDA, CUDA.CUSPARSE
using Printf, PrettyTables

include("../common/common.jl")
import .Common: Formatter, fmt, bold, cyan, blue, green, red, yellow, @addtime!

include("ProcessingUnit.jl") 
include("ReOrthTechnics/ReOrthTechnics.jl")
include("OrthonormalBasis/OrthonormalBasis.jl")

include("LanczosIterator.jl")
include("Factorization/Factorization.jl")
include("Convergence/Convergence.jl")
include("FactorizationReport.jl")

include("LanczosAlgorithm.jl")
include("LanczosMethod.jl")


"""
    lanczos(mat::AbstractMatrix, x0::AbstractVecOrMat, howmany::Int; 
            basistype::Type{<:OrthonormalBasis}=MatrixBasis, 
            rot::ReOrthTechnique=FullRO(), 
            maxdim::Int=10howmany, 
            which::Symbol=:SR, 
            tol::Real=1e-14, 
            eigentol::Real=1e-8, 
            mapvals::Union{Function,Nothing}=nothing)

    lanczos(f!::Function, x0::AbstractVecOrMat, howmany::Int; 
            basistype::Type{<:OrthonormalBasis}=MatrixBasis, 
            rot::ReOrthTechnique=FullRO(), 
            maxdim::Int=10howmany, 
            which::Symbol=:SR, 
            tol::Real=1e-14, 
            eigentol::Real=1e-8, 
            mapvals::Function=f!)

Performs the **block Lanczos factorization** to compute a specified number of extremal eigenvalues and eigenvectors of a matrix or a linear operator.  
This method automatically detects whether the input data lives on the CPU or GPU, and dispatches to the appropriate optimized backend.

# Arguments
- `mat::AbstractMatrix`: The input matrix for which the spectral decomposition is sought.
- `f!::Function`: Matrix–vector multiplication routine, defined as `f!(Y, X) = A * X`.  
  Used when the matrix is represented implicitly or stored on GPU.
- `x0::AbstractVecOrMat`: Initial block of orthonormal starting vectors.  
  The number of columns determines the block size.
- `howmany::Int`: Number of desired eigenpairs.

# Keyword Arguments
- `basistype::Type{<:OrthonormalBasis}=MatrixBasis`: Type of orthonormal basis used (`MatrixBasis`, `HybridMatrixBasis`, or `VectorBasis`).
- `maxdim::Int=500howmany`: Maximum dimension of the Krylov subspace.
- `which::Symbol=:SR`: Determines which eigenvalues to converge (`:SR`, `:LR`, or `:amplitude`).
- `tol::Real=1e-14`: Numerical tolerance for orthogonality loss in the Lanczos basis.
- `eigentol::Real=1e-8`: Convergence tolerance for eigenvalue residuals.
- `mapvals::Union{Function,Nothing}=nothing`: Optional mapping function applied to the spectral transformation.  
  Defaults to standard matrix–vector multiplication.

# Description
The Lanczos algorithm is a Krylov subspace method used to approximate the smallest (or largest) eigenvalues 
and corresponding eigenvectors of large Hermitian or symmetric operators.  

Two calling conventions are supported:
1. **Matrix interface:** Pass a concrete matrix `mat` — a multiplication function `f!` is generated internally.  
2. **Operator interface:** Pass a user-defined function `f!(Y, X)` for custom or GPU-resident operators.

The function constructs:
- A `LanczosIterator` to manage Krylov basis vectors and reorthogonalization.
- A `ConvergenceInfo` object to track eigenvalue convergence and stopping criteria.
- A `FactorizationReport` for diagnostic output and benchmarking.

GPU arrays (`CuArray`) are automatically detected, and computations are performed on the GPU via CUDA.

# Returns
A tuple containing:
1. `vals`: Approximated eigenvalues (sorted according to `which`).
2. `vecs`: Corresponding approximate eigenvectors (as columns).
3. `report::FactorizationReport`: Detailed factorization report containing timing, convergence, and iteration statistics.

"""
function lanczos(
    mat::AbstractMatrix, x0::AbstractVecOrMat, howmany::Int;
    basistype::Type{<:OrthonormalBasis} = MatrixBasis,
    rot::ReOrthTechnique                = FullRO(),
    maxdim::Int                         = 500howmany,
    which::Symbol                       = :SR,
    tol::Real                           = 1e-14,
    eigentol::Real                      = 1e-8,
    mapvals::Union{Function,Nothing}    = nothing
)
    f!(Y::AbstractVecOrMat, X::AbstractVecOrMat) = mul!(Y,  mat, X)
    mapvals = isnothing(mapvals) ? f! : mapvals

    return lanczos(
        f!, x0, howmany;
        basistype=basistype,
        rot=rot,
        maxdim=maxdim,
        which=which,
        tol=tol,
        eigentol=eigentol,
        mapvals=mapvals
    )
end



function lanczos(
    f!::Function, x0::AbstractVecOrMat, howmany::Int; 
    basistype::Type{<:OrthonormalBasis} = MatrixBasis, 
    rot::ReOrthTechnique                = FullRO(), 
    maxdim::Int                         = 500howmany, 
    which::Symbol                       = :SR,
    tol::Real                           = 1e-14, 
    eigentol::Real                      = 1e-8, 
    mapvals::Function                   = f!
)

    pu = isa(x0, CuArray) ? GPU() : CPU() #determine wether to use GPU or not
    
    blocksize = size(x0, 2)
    maxiter = ceil(Int, maxdim/blocksize) # ensure that maxdim is devideble with s
    maxdim = maxiter * blocksize
    
    iterator    = LanczosIterator(f!, x0, rot)
    convergence = ConvergenceInfo(howmany, blocksize, maxiter, tol, eigentol, which, mapvals)

    lanczos_method(iterator, convergence, basistype, pu)
end


export FullRO, PartialRO, ReOrthTechnique
export MatrixBasis, HybridMatrixBasis, VectorBasis
export GPU, CPU
export lanczos
export FactorizationReport
export EigSorter
export display_factorization_report, print_factorization_report

end


# using SparseArrays, LinearAlgebra, CUDA
# # include("../Lanczos_old/Lanczos.jl")
            
# function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
#     basis = [b for b in 0:2^L-1 if count_ones(b) == Nup] # generate basis
#     dim = length(basis)
#     bmap = Dict(b => i for (i, b) in enumerate(basis))  # state index map
#     rows, cols, vals = Int[], Int[], Float64[]
#     for (col, state) in enumerate(basis)
#         for i in 1:L
#             j = i % L + 1  # PBC
#             si = (state >> (i - 1)) & 1
#             sj = (state >> (j - 1)) & 1
#             # --- S^z_i S^z_j diagonal term ---
#             SzSz = (0.5 - si) * (0.5 - sj)  # spin-½: Sz = ±½
#             push!(rows, col); push!(cols, col); push!(vals, delta * SzSz)
#             # --- S⁺_i S⁻_j + h.c. (flip-flop term) ---
#             if si != sj
#                 flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
#                 if haskey(bmap, flipped) 
#                     push!(rows, bmap[flipped]); push!(cols, col); push!(vals, 0.5)
#                 end
#             end
#         end
#     end
#     return sparse(rows, cols, vals, dim, dim)
# end




# function test_lanczos() 

#     L =14
#     mat = construct_xxz_spin_sector(L, 0.123, Int(L÷2)) # XXZ model with delta=0.0

#     display(mat)
#     D = size(mat, 1)

#     # x0_ = rand(D)
#     # x0 = x0_ ./ norm(x0_)

#     x0_ = rand(D,4)
#     x0 = Matrix(qr(x0_).Q) # orthonormalize
#     howmany = 10

#     vals, vecs = Lanczos.lanczos(mat, x0, howmany; maxdim=1000, tol=1e-14, eigentol=1e-10)
#     # println("Computed eigenvalues:\n", vals)
#     # println("Corresponding eigenvectors:\n")
#     # println(vecs)
# end


# # function test_lanczos_CUDA() 
# #     L =20
# #     mat = construct_xxz_spin_sector(L, 0.123, Int(L÷2)) 
# #     mat_cu = CUDA.CUSPARSE.CuSparseMatrixCSR(mat) # convert to CuMatrix for GPU
# #     f!(Y,X) = mul!(Y, mat_cu, X)
# #     D = size(mat, 1)

# #     x0_ = CUDA.rand(Float64, D,4)
# #     x0 = CuMatrix(qr(x0_).Q) # orthonormalize
# #     howmany = 2

# #     vals, vecs, factorization_report = Lanczos.lanczos(f!, x0, howmany; maxdim=1000, tol=1e-14, eigentol=1e-8, basistype=Lanczos.HybridMatrixBasis)
# #     Lanczos.display_report(factorization_report)


# #     vals, vecs, factorization_report = Lanczos.lanczos(f!, x0, howmany; maxdim=1000, tol=1e-14, eigentol=1e-8, basistype=Lanczos.HybridMatrixBasis)
# #     Lanczos.display_report(factorization_report)

# #     # vals_true, vecs_true = Lanczos2.lanczosmethod(f!, x0, howmany; maxdim = 1000, tol = 1e-14, eigentol = 1e-8)

# #     # println(Vector(vals_true) ≈ Vector(vals))
# #     # println(Matrix(vecs_true) ≈ Matrix(vecs))

# #     # errs = abs.(Matrix(vecs_true) - Matrix(vecs))


# #     # # println(errs)
# #     # for col in eachcol(errs)
# #     #     println("Error norm for column: ", norm(col))
# #     #     println("Max error: ", maximum(col))
# #     # end
# #     # println("Max error (norm): ", maximum(errs))
# #     # println("Max error (norm): ", norm(errs))
# # end


# # test_lanczos_CUDA()




