

include("Structs/Structs.jl")
include("PolfedDefaults.jl")
include("polfed_algorithm.jl")
include("DensetiesOfStates/DensetiesOfStates.jl")
include("SpectralTransformation/SpectralTransformation.jl")
include("Optimization/optimization.jl")
include("workers.jl")



"""
    polfed(mat, x0, howmany, target; kwargs...)

Solve an eigenvalue problem using the **Polynomial Filtering Eigenvalue Decomposition (POLFED)** algorithm with a given matrix input.

This method employs Chebyshev polynomial spectral filtering to efficiently extract eigenvalues and eigenvectors within a specified spectral region of a dense or sparse matrix.  
When `optimize_mapping = true`, the operator is internally rescaled and optimized to reduce memory access, particularly effective for Hamiltonians with a small number of unique off-diagonal elements.

# Arguments
- `mat::AbstractMatrix`: The Hamiltonian or operator matrix (dense or sparse).
- `x0::AbstractVecOrMat`: The initial vector or block of vectors used to start the iteration.  
  If a vector is provided, a standard Lanczos factorization is performed; if a matrix is provided, the Block Lanczos variant is used.
- `howmany::Integer`: Number of eigenvalues to compute.
- `target::Union{Real, Nothing}`: Spectral region to target (in unscaled units).  
  If `nothing`, the algorithm automatically targets the region with the highest density of states.

# Keyword Arguments
- `produce_report::Bool = PolfedDefaults.produce_report`  
  Whether to return detailed diagnostic information about the run.  
  If enabled, a [`Report`](@ref) struct is returned alongside eigenvalues and eigenvectors.
- `optimize_mapping::Bool = PolfedDefaults.optimize_mapping`  
  Whether to optimize the spectral mapping for structured Hamiltonians (those with few unique off-diagonal elements).  
  When enabled, a specialized mapping is constructed to minimize memory access, and the corresponding rescaled mapping and Clenshaw recurrence relations are automatically optimized.
- `spectral_transform::SpectralTransformConfig = SpectralTransformConfig()`  
  Configuration for the polynomial spectral transformation (see [`SpectralTransformConfig`](@ref)).  
  This allows control over the polynomial coefficients, normalization, polynomial order, and other parameters.
- `fact::FactorizationConfig = FactorizationConfig()`  
  Configuration for the underlying Lanczos/Krylov factorization (see [`FactorizationConfig`](@ref)).  
  Includes settings for numerical tolerance, reorthogonalization strategy, and eigenvalue convergence criteria.
- `dos::DoSConfig = DoSConfig()`  
  Configuration for the stochastic density-of-states (DoS) estimation (see [`DoSConfig`](@ref)).  
  Parameters include the number of random vectors, number of Chebyshev moments, and statistical averaging options.

# Returns
- `(vals, vecs)` — Eigenvalues and eigenvectors.  
- `(vals, vecs, report)` — If `produce_report = true`, also returns a [`Report`](@ref) containing:
  - Spectral transform diagnostics (mapping, scaling, polynomial order)
  - Factorization statistics (iterations, reorthogonalization)
  - Benchmark metrics (wall time, CPU time, and device information)

# Notes
- The Chebyshev polynomial order is chosen automatically unless explicitly set via `SpectralTransformConfig.order`.
- When `optimize_mapping = true`, a rescaled operator and optimized Clenshaw recurrence are used to reduce memory bandwidth usage, especially beneficial for structured Hamiltonians.
- GPU execution is automatically enabled when `x0 isa CuArray`.


# Examples
```julia
using Polfed
```
"""
function polfed(mat::AbstractMatrix{T}, x0::AbstractVecOrMat{T}, howmany::Integer, target::Union{Real,Nothing};
    produce_report::Bool    = PolfedDefaults.produce_report,
    optimize_mapping::Bool  = PolfedDefaults.optimize_mapping,
    spectral_transform      = SpectralTransformConfig(),
    fact                    = FactorizationConfig(),
    dos                     = DoSConfig(),
) where {T<:Real}


    f! = (Y,X) -> mul!(Y, mat, X)
    optimize_mapping && (f! = optimize_spectral_transform(mat, spectral_transform))

    polfed(f!, x0, howmany, target; 
        produce_report      = produce_report,
        spectral_transform  = spectral_transform,
        fact                = fact,
        dos                 = dos
    )
end


function polfed(f!::Function, x0::AbstractVecOrMat{T}, howmany::Integer, target::Union{Real,Nothing};
    produce_report::Bool    = PolfedDefaults.produce_report,
    spectral_transform      = SpectralTransformConfig(),
    fact                    = FactorizationConfig(),
    dos                     = DoSConfig(),
) where {T<:Real}
    set_workers(x0, spectral_transform.parallelization)

    walltime = zeros(Float64, 1)
    cputime = zeros(Float64, 1)
    @addtime! walltime cputime 1 begin
        pu = isa(x0, CuArray) ? GPU() : CPU()

        spectral_transform_config = SpectralTransformConfigFull(spectral_transform, f!, x0, howmany, target, pu)
        fact_config = FactorizationConfigFull(fact, spectral_transform_config, x0, howmany)
        dos_config = DoSConfigFull(dos)

        vals, vecs, factorization_report = polfed_algorithm(spectral_transform_config, fact_config, dos_config, pu)

        nothing
    end

    remove_workers(spectral_transform.parallelization)

    spectral_transform_report = SpectralTransformReport(spectral_transform_config, factorization_report)
    benchmark_report = BenchmarkReport(factorization_report, walltime[1], cputime[1], x0, pu)
    report = Report(spectral_transform_report, factorization_report, benchmark_report)

    produce_report && (return (vals, vecs, report))
    return (vals, vecs)
end

