"Fast and precise calculation of matrix polynomial expansion"
module ClenshawMapping

using LinearAlgebra, SparseArrays, StaticArrays


polynomial_properties = Dict(
    :Chebyshev =>  ((k)->2           , (k)->-1      , 1, 1),
    :Legendre  =>  ((k)->(2k+1)/(k+1), (k)->-k/(k+1), 1, 1),
    :Hermite   =>  ((k)->2           , (k)->-2*k    , 1, 2),
    :Taylor    =>  ((k)->1           , (k)->0       , 1, 1),
)


include("Algorithm/FinalSum.jl")
include("Algorithm/RecurrenceRelations.jl")
include("Algorithm/ClenshawAlgorithm.jl")
include("Clenshaw.jl")


export Clenshaw
end
