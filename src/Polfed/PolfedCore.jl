module PolfedCore

using Distributed
using LinearAlgebra, UnPack, Base.Threads, QuadGK, Printf, SharedArrays, SparseArrays, Logging
using StaticArrays: SVector

import ..Common: Formatter, fmt, bold, cyan, blue, green, red, yellow, @addtime!
import ..ClenshawMapping
import ..ClenshawMapping: Clenshaw
import ..Lanczos: lanczos, FactorizationReport, display_factorization_report,
        FullRO, PartialRO, ReOrthTechnique,
        MatrixBasis, HybridMatrixBasis, OrthonormalBasis
import ..CuArray, ..CuVector, ..CuMatrix, ..CUDA_AVAILABLE, ..is_gpu_array, ..main_module_file
import ..CUDA

include("Structs/Structs.jl")
include("Structs/PolfedDefaults.jl")
include("Plan/Plan.jl")
include(joinpath(@__DIR__, "..", "Optimization", "optimization.jl"))
include(joinpath(@__DIR__, "..", "DensetiesOfStates", "DensitiesOfStates.jl"))
include("Algorithm/SpectralTransformation/SpectralTransformation.jl")
include("Algorithm/polfed_algorithm.jl")
include("Plan/workers.jl")



"""
    polfed(mat, x0, howmany, target; kwargs...)
    polfed(f!, x0, howmany, target; kwargs...)

Solve an eigenvalue problem using the **Polynomial Filtering Eigenvalue Decomposition (POLFED)** algorithm with a given matrix input.

This method employs Chebyshev polynomial spectral filtering to efficiently extract eigenvalues and eigenvectors within a specified spectral region of a dense or sparse matrix.  
When `optimize_mapping = true`, the operator is internally rescaled and optimized to reduce memory access, particularly effective for Hamiltonians with a small number of unique off-diagonal elements.

# Arguments
- `mat::AbstractMatrix`: The Hamiltonian or operator matrix (dense or sparse).
- `f!::Function`: In-place linear operator callback with signature `f!(Y, X)`.
- `x0::AbstractVecOrMat`: The initial vector or block of vectors used to start the iteration.  
  If a vector is provided, a standard Lanczos factorization is performed; if a matrix is provided, the Block Lanczos variant is used.
- `howmany::Integer`: Number of eigenvalues to compute.
- `target`: Spectral region to target (in unscaled units).  
  Accepts a numeric value (treated as unrescaled) or a spec such as
  `:maxdos`, `:middle`, `(:offset, frac)`,
  `(:unrescaled, value)`, or `(:rescaled, value)`.

# Keyword Arguments
- `produce_report::Bool = PolfedDefaults.produce_report`  
  Whether to return detailed diagnostic information about the run.  
  If enabled, a [`Report`](@ref) struct is returned alongside eigenvalues and eigenvectors.
- `mapping::MappingConfig = MappingConfig()`  
  Configuration for mapping, rescaling, parallelization, and optional optimized kernels.
- `transform::TransformConfig = TransformConfig()`  
  Configuration for polynomial coefficients, normalization, interval, and order.
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
- The polynomial order is chosen automatically unless explicitly set via `TransformConfig.order`.
- When `mapping.optimize_mapping = true`, a rescaled operator and optimized Clenshaw recurrence are used to reduce memory bandwidth usage, especially beneficial for structured Hamiltonians.
- GPU execution is automatically enabled when `is_gpu_array(x0)`.


# See also
[`MappingConfig`](@ref), [`TransformConfig`](@ref), [`FactorizationConfig`](@ref), [`DoSConfig`](@ref), [`display_report`](@ref)
"""
function polfed(mat::AbstractMatrix{T}, x0::AbstractVecOrMat{T}, howmany::Integer, target;
    produce_report::Bool             = PolfedDefaults.produce_report,
    mapping::MappingConfig           = MappingConfig(),
    transform::TransformConfig       = TransformConfig(),
    fact                             = FactorizationConfig(),
    dos                              = DoSConfig(),
) where {T<:Number}
    if is_gpu_array(x0) && !(eltype(x0) <: Real)
        error("Complex GPU arrays are not supported yet. Current GPU kernels are real-only. Use CPU arrays for complex matrices/operators.")
    end

    PolfedDefaults.polfed_log(
        PolfedDefaults.POLFED_INFO_LEVEL,
        "Starting POLFED (matrix input).",
        howmany=howmany,
        target=target,
    )

    f! = (Y,X) -> mul!(Y, mat, X)
    if mapping.optimize_mapping
        f! = optimize_spectral_transform(mat, mapping)
    end

    polfed(f!, x0, howmany, target;
        produce_report  = produce_report,
        mapping         = mapping,
        transform       = transform,
        fact            = fact,
        dos             = dos
    )
end


"""
    polfed(f!::Function, x0::AbstractVecOrMat, howmany::Integer, target; kwargs...)

Operator-callback overload of [`polfed`](@ref).

Use this when the matrix is represented implicitly through `f!(Y, X)`.
"""
function polfed(f!::Function, x0::AbstractVecOrMat{T}, howmany::Integer, target;
    produce_report::Bool             = PolfedDefaults.produce_report,
    mapping::MappingConfig           = MappingConfig(),
    transform::TransformConfig       = TransformConfig(),
    fact                             = FactorizationConfig(),
    dos                              = DoSConfig(),
) where {T<:Number}
    walltime = zeros(Float64, 1)
    cputime = zeros(Float64, 1)
    parallel_strategy = nothing
    workers_set = false

    local vals
    local vecs
    local factorization_report
    local mapping_plan
    local transform_plan
    local pu

    try
        @addtime! walltime cputime 1 begin
            if mapping.optimize_mapping && isnothing(mapping.f!_rescaled)
                PolfedDefaults.polfed_log(
                    PolfedDefaults.POLFED_WARN_LEVEL,
                    "optimize_mapping is ignored for operator input; provide a matrix to enable mapping optimization.",
                )
            end

            pu = is_gpu_array(x0) ? GPU() : CPU()
            if isa(pu, GPU) && !(eltype(x0) <: Real)
                error("Complex GPU arrays are not supported yet. Current GPU kernels are real-only. Use CPU arrays for complex operators.")
            end

            PolfedDefaults.polfed_log(
                PolfedDefaults.POLFED_INFO_LEVEL,
                "Starting POLFED (operator input).",
                howmany=howmany,
                target=target,
                processing_unit=typeof(pu),
            )

            mapping_plan = build_mapping_plan(mapping, f!, x0, pu)
            parallel_strategy = mapping_plan.parallel_strategy
            set_workers(x0, parallel_strategy)
            workers_set = true
            
            transform_plan = build_transform_plan(transform, mapping_plan, howmany, target)
            fact_config = FactorizationConfigFull(fact, x0, howmany)
            dos_config = DoSConfigFull(dos)

            vals, vecs, factorization_report = polfed_algorithm(transform_plan, mapping_plan, fact_config, dos_config, pu)

            nothing
        end
    finally
        workers_set && remove_workers(parallel_strategy)
    end

    spectral_transform_report = SpectralTransformReport(transform_plan, mapping_plan, factorization_report)
    benchmark_report = BenchmarkReport(factorization_report, walltime[1], cputime[1], x0, pu)
    report = Report(spectral_transform_report, factorization_report, benchmark_report)

    produce_report && (return (vals, vecs, report))
    return (vals, vecs)
end

end # module
