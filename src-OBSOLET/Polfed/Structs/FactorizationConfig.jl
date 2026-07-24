"""
    FactorizationConfig(; kwargs...)

Configuration structure for the Krylov factorization used within POLFED.

This struct defines options that control the iterative factorization process, 
including re-orthogonalization strategies, basis representation, and convergence tolerances.  
These parameters are passed to the underlying Lanczos or Block Lanczos solver.

# Fields
- `rot::ReOrthTechnique`  
  Re-orthogonalization technique. Options:
  - `FullRO()` (default): Full re-orthogonalization.
  - `PRO()`: Partial re-orthogonalization.  
  Since the polynomial spectral transformation typically dominates the runtime,
  the overhead of full re-orthogonalization is negligible, making `FullRO()` generally preferred.

- `basistype::Type{<:OrthonormalBasis}`  
  Type of orthonormal basis used. Default: [`PolfedDefaults.basistype`](@ref).  
  Options:
  - `MatrixBasis`: Standard CPU-based orthonormal basis.
  - `HybridMatrixBasis`: Enables hybrid CPU–GPU computation (slower due to data transfer overhead).

- `which::Symbol`  
  Specifies which eigenvalues to target (`:LR` or `:SR`).  
  Default: [`PolfedDefaults.which`](@ref).

- `tol::Real`  
  Convergence tolerance for the Krylov factorization.  
  Default: [`PolfedDefaults.tol`](@ref).  
  Convergence is assessed based on the residual norm of the Ritz pairs.

- `eigentol::Union{Real,Nothing}`  
  Convergence tolerance for eigenvectors.  
  Default: [`PolfedDefaults.eigentol`](@ref).  
  Computed using the L₂ norm of the eigenvector residual.

# Example

```julia
fact_cfg = FactorizationConfig(tol=1e-12, which=:SR)
```
# Notes

- The tol parameter typically controls when Ritz values are accepted as converged.
- For most applications, full re-orthogonalization is recommended for numerical stability.
- The hybrid basis is mainly useful when the matrix–vector products are performed on the GPU.
"""
mutable struct FactorizationConfig
    rot::ReOrthTechnique
    basistype::Type{<:OrthonormalBasis}
    which::Symbol
    tol::Real
    eigentol::Union{Real,Nothing}
    overestimate_iters::Real


    """Build a `FactorizationConfig` from keyword arguments."""
    function FactorizationConfig(;
        rot=PolfedDefaults.rot, 
        basistype::Type{<:OrthonormalBasis}=PolfedDefaults.basistype,
        which::Symbol=PolfedDefaults.which,
        tol::Real=PolfedDefaults.tol, 
        eigentol::Union{Real, Nothing}=PolfedDefaults.eigentol, 
        overestimate_iters::Real=PolfedDefaults.overestimate_iters,
    )
        new(rot, basistype, which, tol, eigentol, overestimate_iters)
    end

end


"""
`FactorizationConfigFull`

Fully-resolved configuration for a Krylov factorization (e.g., Lanczos)
used in POLFED, including vector dimensions and the computed maximum
Krylov subspace size.

# Fields

- `x0::AbstractVecOrMat` — Initial vector(s) for the factorization.
- `elmtype::Type` — Element type of `x0` (usually `Float64` or `Float32`).
- `maxdim::Integer` — Maximum Krylov subspace dimension, computed using the formula:
"""
mutable struct FactorizationConfigFull
    x0::AbstractVecOrMat
    elmtype::Type
    maxdim::Integer
    rot::ReOrthTechnique
    basistype::Type{<:OrthonormalBasis}
    which::Symbol
    tol::Real
    eigentol::Union{Real,Nothing}
    overestimate_iters::Real


    """Build resolved `FactorizationConfigFull` from user config and runtime shape."""
    function FactorizationConfigFull(
        fact::FactorizationConfig,
        x0::AbstractVecOrMat{T},
        howmany::Integer,
    ) where {T<:Number}
    
        blocksize = size(x0,2)
        
        new(
            x0,
            T,
            PolfedDefaults.expectedkrylovdim(howmany, blocksize, fact.overestimate_iters),
            fact.rot,
            fact.basistype,
            fact.which,
            fact.tol,
            fact.eigentol,
            fact.overestimate_iters,
        )
    end

end
