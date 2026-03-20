module Lanczos

using LinearAlgebra, Printf, PrettyTables, Logging
import ..CUDA_AVAILABLE, ..CuArray, ..CuVector, ..CuMatrix, ..is_gpu_array
import ..CUDA

import ..Common: Formatter, fmt, bold, cyan, blue, green, red, yellow, @addtime!,
                 POLFED_SILENT_LEVEL, POLFED_WARN_LEVEL, POLFED_INFO_LEVEL, POLFED_DEBUG_LEVEL,
                 verbosity, should_log, polfed_log

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

GPU arrays are automatically detected via `is_gpu_array`, and computations are performed on the GPU via CUDA.

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



"""
    lanczos(f!::Function, x0::AbstractVecOrMat, howmany::Int; kwargs...)

Operator-callback overload of [`lanczos`](@ref).
"""
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

    pu = is_gpu_array(x0) ? GPU() : CPU() #determine wether to use GPU or not
    if isa(pu, GPU) && !(eltype(x0) <: Real)
        error("Complex GPU arrays are not supported yet. Current GPU Lanczos path is real-only. Use CPU arrays for complex operators.")
    end
    
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
