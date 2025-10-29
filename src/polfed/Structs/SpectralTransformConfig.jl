"""
    SpectralTransformConfig(; kwargs...)

Configuration structure for spectral transformations in POLFED.

This type stores all settings controlling the polynomial-based spectral transformation used in POLFED.
It determines how the Hamiltonian is rescaled, how the polynomial filter is constructed, and how
parallelization and recurrence optimizations are applied.

All fields can be set manually through keyword arguments, or left as `nothing`/defaults to use values from
[`PolfedDefaults`](@ref).

# Fields
- `coefficients::Function`:  
  Function returning polynomial coefficients. Must accept two arguments:
  `(target::T, n::Int) where {T<:Real}`, and return the coefficient of the `n`-th order polynomial evaluated at `target`.

- `normalization::Real`:  
  Normalization factor of the polynomial. Defines the value to which `P_K(target)` is normalized.

- `cutoff::Real`:  
  Threshold above which eigenvalues of the transformed Hamiltonian are the largest within the targeted interval.

- `left::Union{Real,Nothing}` / `right::Union{Real,Nothing}`:  
  Optional spectral interval boundaries defining the target energy window.  
  If `nothing`, the interval is inferred automatically.

- `order::Union{Int,Nothing}`:  
  Polynomial order of the spectral transformation.  
  If `nothing`, the order is determined automatically from the target energy or interval.

- `overestimate_iters::Real`:  
  Safety factor used when estimating the required Krylov subspace size.  
  The expected dimension is predicted via [`PolfedDefaults.expectedkrylovdim`](@ref) and multiplied by this factor to ensure convergence.  
  Default: [`PolfedDefaults.overestimate_iters`](@ref).

- `order_safety_factor::Real`:  
  Factor applied to slightly reduce the polynomial order, ensuring that all `howmany` targeted eigenvalues are captured even if the estimated density of states is imperfect.

- `parallelization::Union{Parallelization,Nothing}`:  
  Parallelization strategy for matrix–vector operations.  
  Available strategies are [`NoParallel`](@ref), [`MulColsParallel`](@ref), and [`TwoLevelParallel`](@ref) (see also [`Parallelization`](@ref)).  
  If `nothing`, a default strategy is chosen based on the processing unit (`MulColsParallel` for CPU, `NoParallel` for GPU).

- `f!_rescaled::Union{Nothing,Function}`:  
  Optional rescaled operator function.  
  Should have the signature `(Y::AbstractVecOrMat, X::AbstractVecOrMat)` and perform an in-place operation equivalent to `@. Y = H̃ * X`, where `H̃` is the rescaled Hamiltonian.

- `clenshaw_recurrence::Union{Nothing,Function}`:  
  Optional Clenshaw recurrence kernel to reduce memory access.  
  Should have the signature `(b1::AbstractVector, b2::AbstractVector, b3::AbstractVector, c::Real, X::AbstractVector)`, performing an update equivalent to `@. b1 = 2c * H̃ * b2 - b3 + c * X`.

- `clenshaw_finalsum::Union{Nothing,Function}`:  
  Optional final summation function for the Clenshaw recurrence.  
  Should have the signature `(b1::AbstractVector, b2::AbstractVector, c::Real, Y::AbstractVector, X::AbstractVector)`, performing an operation like `@. Y = c * X + H̃ * b1 - b2`.

- `Emin::Union{Real,Nothing}` / `Emax::Union{Real,Nothing}`:  
  Optional lower and upper spectral bounds.  
  If left as `nothing`, they are automatically estimated using the Lanczos algorithm.

# Constructor
```julia
SpectralTransformConfig(; kwargs...)
```
All keyword arguments correspond directly to the fields listed above.
If a field is omitted, its default value is taken from PolfedDefaults.

# Notes
- This struct is typically passed as the spectral_transform argument to polfed.
- Custom mappings and recurrence kernels can be supplied for specialized structured Hamiltonians.
"""
mutable struct SpectralTransformConfig
    coefficients::Function
    normalization::Real
    cutoff::Real
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    order::Union{Int, Nothing}
    order_safety_factor::Real
    parallelization::Union{Parallelization,Nothing}
    f!_rescaled::Union{Nothing, Function}   
    clenshaw_recurrence::Union{Nothing, Function}
    clenshaw_finalsum::Union{Nothing, Function}
    overestimate_iters::Real
    Emin::Union{Real, Nothing}
    Emax::Union{Real, Nothing}

    function SpectralTransformConfig(; 
        coefficients::Function                          =   PolfedDefaults.coefficients, 
        normalization::Real                             =   PolfedDefaults.normalization, 
        cutoff::Real                                    =   PolfedDefaults.cutoff, 
        left::Union{Real, Nothing}                      =   nothing,
        right::Union{Real, Nothing}                     =   nothing,
        order::Union{Integer, Nothing}                  =   nothing,
        order_safety_factor::Real                       =   PolfedDefaults.order_safety_factor,
        parallelization::Union{Parallelization,Nothing} =   PolfedDefaults.parallelization,
        f!_rescaled::Union{Nothing, Function}           =   nothing,
        clenshaw_recurrence::Union{Nothing, Function}   =   nothing,
        clenshaw_finalsum::Union{Nothing, Function}     =   nothing,
        overestimate_iters::Real                        =   PolfedDefaults.overestimate_iters,
        Emin::Union{Real, Nothing}                      =   nothing,
        Emax::Union{Real, Nothing}                      =   nothing,
    )
        new(coefficients, normalization, cutoff, left, right, order, order_safety_factor, parallelization, f!_rescaled, clenshaw_recurrence, clenshaw_finalsum, overestimate_iters, Emin, Emax)
    end
end



mutable struct SpectralTransformConfigFull
    coefficients::Function
    normalization::Real
    polynomialtype::Symbol
    cutoff::Real
    target::Union{Real, Nothing}
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    howmany::Integer
    order::Union{Integer, Nothing}
    order_safety_factor::Real
    parallelization::Parallelization
    f!::Function
    f!_rescaled::Union{Function,Nothing}
    f!_transformed::Union{Function,Nothing}
    clenshaw_recurrence::Union{Nothing, Function}
    clenshaw_finalsum::Union{Nothing, Function}
    overestimate_iters::Real
    Emin::Real
    Emax::Real


    function SpectralTransformConfigFull(
        cfg::SpectralTransformConfig,
        f!::Function,
        x0::AbstractVecOrMat{T},
        howmany::Integer,
        target::Union{Real, Nothing},
        pu::ProcessingUnit,
    ) where {T<:Real}
        parallelization = nothing
        if isa(cfg.parallelization, Nothing)
            isa(pu, CPU) && (parallelization = MulColsParallel())
            isa(pu, GPU) && (parallelization = NoParallel())
        else
            parallelization = cfg.parallelization
        end

        v0 = pu.Vector(x0[:,1])        
        Emin = isnothing(cfg.Emin) ? first(collect(lanczos(f!, v0, 1; which=:SR, maxdim=1000)[1])) : cfg.Emin
        Emax = isnothing(cfg.Emax) ? last(collect(lanczos(f!, v0, 1; which=:LR,  maxdim=1000)[1])) : cfg.Emax

        a = (Emax-Emin)/2
        b = (Emax+Emin)/2

        f!_rescaled = !isa(cfg.f!_rescaled, Nothing) ? cfg.f!_rescaled : f!_rescaled_fun(Y::AbstractVecOrMat,X::AbstractVecOrMat) = begin
            f!(Y,X) 
            @. Y *= 1/a
            @. Y -= (b/a)*X
        end

        target_rescale  = isa(target, Nothing)      ? nothing : (target - b) / a
        left_rescale    = isa(cfg.left, Nothing)    ? nothing : (cfg.left - b) / a
        right_rescale   = isa(cfg.right, Nothing)   ? nothing : (cfg.right - b) / a


        new(
            cfg.coefficients,
            cfg.normalization,
            PolfedDefaults.polynomialtype,
            cfg.cutoff,
            target_rescale,
            left_rescale,
            right_rescale,
            howmany,
            cfg.order,
            cfg.order_safety_factor,
            parallelization,
            f!,
            f!_rescaled,
            nothing,
            cfg.clenshaw_recurrence,
            cfg.clenshaw_finalsum,
            cfg.overestimate_iters,
            Emin,
            Emax,
        )
    end
end



