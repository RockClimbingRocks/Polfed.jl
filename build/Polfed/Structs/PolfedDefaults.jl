"""
`PolfedDefaults` — Default settings for POLFED.jl

This module contains all default constants and functions used across the POLFED
framework, including:

- Krylov subspace dimension prediction
- Reporting and mapping options
- Spectral transformation defaults (polynomial type, coefficients, cutoff, etc.)
- Lanczos algorithm defaults (rotation type, basis, tolerances)
- Density-of-states calculation defaults (kernel, number of points, random vectors)

All defaults are intended to provide reasonable starting points for typical
simulations, but can be overridden by user-specified options.

Default values:

- `produce_report = false`
- `optimize_mapping = false`
- `polynomialtype = :Chebyshev`
- `cutoff = 0.17`
- `normalization = 1.0`
- `order_safety_factor = 0.97`
- `parallel_strategy = MulColsParallel()`
- `overestimate_iters = 1.25`
- `rot = FullRO()`
- `basistype = MatrixBasis`
- `tol = 1e-14`
- `eigentol = 1e-9`
- `which = :LR`
- `kernel = :Jackson`
- `N = 250`
- `R = 300`
"""
module PolfedDefaults
import ..MulColsParallel, ..FullRO, ..MatrixBasis
import ...Common: POLFED_SILENT_LEVEL, POLFED_WARN_LEVEL, POLFED_INFO_LEVEL, POLFED_DEBUG_LEVEL,
                  verbosity, should_log, polfed_log

##############################
# Krylov Dimension Prediction
##############################

"""
`expectedkrylovdim(howmany::Int, blocksize::Int, η::Real)`

Predicts the expected Krylov subspace dimension for POLFED
given the number of vectors (`howmany`), block size (`blocksize`),
and a safety factor `η`.

Returns an integer, rounded up using `ceil(Int64, ...)`
```julia
expectedkrylovdim(howmany, blocksize, η) = ceil(Int64, (20.427*blocksize + 1.696*howmany)*η)
```

Example:
```julia
PolfedDefaults.expectedkrylovdim(10, 5, 1.1) # => 240 (example)
```
"""
const expectedkrylovdim(howmany::Int, blocksize::Int, η::Real) =
    ceil(Int64, (20.427*blocksize + 1.696*howmany)*η)


##############################
# Reporting Options
##############################

"""
produce_report::Bool = false

Whether to produce a summary report at the end of a POLFED run.
"""
const produce_report = false

"""
optimize_mapping::Bool = false

Whether to optimize the Hamiltonian mapping before computation. That is beneficial if the Hamiltonian has one or two different offdiagonal values in the matrix.
"""
const optimize_mapping = false

##############################
# Verbosity / Reporting
##############################


##############################
# Spectral Transformation Defaults
##############################

"""
coefficients(λ::T, n::Int) where {T<:Real}

Function that produces polynomial coefficients for spectral transformations.
Default is Chebyshev polynomial:
- For n = 0: coefficient is 2 * cos(0 * acos(λ)) = 2
- For n > 0: coefficient is cos(n * acos(λ))

```julia
const coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n,0)) * cos(n * acos(λ)))
```
"""
const coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n,0)) * cos(n * acos(λ)))

"""
polynomialtype::Symbol = :Chebyshev

Default polynomial type for spectral transformations. Currently only `:Chebyshev` is supported.
"""
const polynomialtype = :Chebyshev


"""
`cutoff::Real = 0.17`

Default cutoff value for the polynomial expansion.
"""
const cutoff = 0.17

"""
`normalization::Real = 1.0`

Spectral transformation is normalized so that P_K(target) = normalization.
"""
const normalization = 1.00

"""
`order_safety_factor::Real=0.97`

Safety factor used to reduce the polynomial order to ensure, all of the `howmany` target eigenvalues are captured within the filter.
"""
const order_safety_factor = 0.97

"""
`parallel_strategy::Parallelization = MulColsParallel()`

Default parallelization strategy for spectral transformations.
Default is `MulColsParallel()`.
"""
const parallel_strategy = MulColsParallel()

"""
`overestimate_iters::Real = 1.25`

Factor used to overestimate the number of iterations needed for convergence of the algorithm. In particular, the predicted Krylov dimension [`expectedkrylovdim`](@ref) is multiplied by this factor to ensure sufficient iterations.

# Note
Consider the case where not all of the requested eigenpairs are converged, and  you check the [`Polfed.Report`](@ref) at the end of the run (with [`display_report`](@ref)). There you can see that not all eigenpairs are converged, and under [`Polfed.Lanczos.FactorizationReport`](@ref) you can see that all iterations were used. This is usually a clear sign that one should increase `overestimate_iters` factor.
"""
const overestimate_iters = 1.25


##############################
# Lanczos Defaults
##############################

"""
`rot::FullRO = FullRO()`

Default reorthogonalization type for the Lanczos algorithm, another option is to use PartialRO(skip) but it does not bring any benefit in most cases.
"""
const rot = FullRO()

"""
`basistype::Type = MatrixBasis`

Default basis type for the Lanczos algorithm, and also the only one.
"""
const basistype = MatrixBasis

"""
`tol::Real = 1e-14`

Default convergence tolerance for Lanczos iterations.
"""
const tol = 1e-14

"""
`eigentol::Real = 1e-9`

Default eigenvalue tolerance for Lanczos eigenvectors.
"""
const eigentol = 1e-9

"""
`which::Symbol = :LR`

Default selection of eigenvalues: `:LR` or `:SR`.
"""
const which = :LR


##############################
# Density of States Defaults
##############################

"""
`kernel::Symbol = :Jackson`

Default kernel used in density-of-states calculations. Available options are also `:Lorentz`, `:Fejer`, `:LanczosK`, `:WangZunger`, and `:Dirichlet`.
"""
const kernel = :Jackson

"""
`N::Int = 250`

Default order of Chebyshev polynomial expansion in density-of-states calculations.
"""
const N = 250

"""
`R::Int = 300`

Default number of random vectors used in density-of-states calculations.
"""
const R = 300

end # module
