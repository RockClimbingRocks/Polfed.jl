
module PolfedDefaults
using ..Polfed


const expectedkrylovdim(howmany::Int, blocksize::Int, η::Real) = ceil(Int64, (20.427*blocksize + 1.696*howmany)*η)


# produce report at the end of the polfed 
const produce_report        = false


# spectral transformation defaults
const coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n,0)) * cos(n * acos(λ)))
const polynomialtype        = :Chebyshev
const cutoff                = 0.17
const normalization         = 1.00
const order_safety_factor   = 0.95
const parallelization       = Polfed.MulColsParallel()
const overestimate_iters    = 1.25


# Lanczos defaults 
const rot       = Polfed.FullRO()
const reorth    = Polfed.MatrixGramSchmidt()
const basistype = Polfed.MatrixBasis
const tol       = 1e-14
const eigentol  = nothing
const which     = :largest


# Denseties of States defaults
const kernel    = :Jackson
const N         = 250
const R         = 300

end