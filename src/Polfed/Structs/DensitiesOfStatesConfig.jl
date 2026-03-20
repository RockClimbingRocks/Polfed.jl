"""
    DoSConfig(; kwargs...)

Configuration structure for Density of States (DoS) calculations in POLFED.

This struct specifies parameters for stochastic estimation of the density of states
using random vector sampling and Chebyshev moment expansion.  
The configuration affects the resolution and statistical accuracy of the DoS computation.

# Fields
- `N::Int`  
  Number of Chebyshev moments (or energy points) used in the DoS calculation.  
  Higher values improve spectral resolution at increased computational cost.  
  Default: [`PolfedDefaults.N`](@ref) (typically `250`).

- `R::Int`  
  Number of random vectors used to stochastically estimate the DoS.  
  Increasing `R` reduces statistical noise in the estimated spectrum.  
  Default: [`PolfedDefaults.R`](@ref) (typically `300`).

- `kernel::Symbol`  
  Kernel used in the KPM expansion.  
  Default: [`PolfedDefaults.kernel`](@ref) (typically `:Jackson`).

# Example

```julia
# Use default DoS settings
dos_cfg = DoSConfig()

# Custom configuration
dos_cfg_custom = DoSConfig(N=500, R=400)

# Notes
- The statistical accuracy of the estimated DoS scales as `1/√R`, while the spectral resolution improves roughly as `1/N`.
- The DoS computation is typically performed as a preparatory step before spectral filtering, to identify the spectral interval around the target energy and determine an appropriate polynomial order.
- For large sparse systems, moderate `R` values (100–400) usually provide a good balance between computational cost and accuracy.
"""
mutable struct DoSConfig
    N::Int
    R::Int
    kernel::Symbol

    function DoSConfig(;
        N::Int=PolfedDefaults.N, 
        R::Int=PolfedDefaults.R,
        kernel::Symbol=PolfedDefaults.kernel
    )
        new(N, R, kernel)   
    end
end


"""
    DoSConfigFull(dos::DoSConfig)

Resolved density-of-states configuration with concrete kernel function and
runtime DoS placeholder `ρ`.
"""
mutable struct DoSConfigFull{K}
    ρ::Union{Function, Nothing}
    kernel::Symbol
    kernel_fn::K
    N::Integer
    R::Integer

    """Build resolved `DoSConfigFull` from `DoSConfig`."""
    function DoSConfigFull(
        dos::DoSConfig
    )
        kernel_fn = get_kernel(dos.kernel)
        new{typeof(kernel_fn)}(nothing, dos.kernel, kernel_fn, dos.N, dos.R)   
    end
end
