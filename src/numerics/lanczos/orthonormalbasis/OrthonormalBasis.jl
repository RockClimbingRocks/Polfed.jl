"""
    OrthonormalBasis

Abstract supertype for Krylov basis storage backends (`MatrixBasis`, `VectorBasis`).
"""
abstract type OrthonormalBasis end

include("MatrixBasis.jl")
include("VectorBasis.jl")

"""
    createbasis(maxdim::Int, x0::AbstractVecOrMat{E}, basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit, layout::Layout=ColumnLayout()) where {E<:Number} -> OrthonormalBasis

Create the concrete basis storage object used by Lanczos iterations. `layout` selects the
block storage orientation (columns for [`ColumnLayout`](@ref), rows for [`RowLayout`](@ref)).
"""
function createbasis(maxdim::Int, x0::AbstractVecOrMat{E},
                     basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit,
                     layout::Layout=ColumnLayout()) where {E<:Number}
    return basistype{E}(maxdim, x0, pu, layout)
end
