abstract type OrthonormalBasis end
include("MatrixBasis.jl")
include("VectorBasis.jl")

abstract type Basis end

struct FullBasis <: Basis
    fullbasis::OrthonormalBasis
    nvecs::Int

    function FullBasis(fullbasis::OrthonormalBasis)
        nvecs = fullbasis.nvecs
        new(fullbasis, nvecs)
    end
end

struct PartialBasis <: Basis
    fullbasis::OrthonormalBasis
    partialbasis::OrthonormalBasis
    nvecs::Int

    function PartialBasis(fullbasis::OrthonormalBasis, partialbasis::OrthonormalBasis)
        nvecs = fullbasis.nvecs + partialbasis.nvecs
        new(fullbasis, partialbasis, nvecs)
    end
end


add!(basis::FullBasis, v::AbstractVecOrMat) = add!(basis.fullbasis, v)
last(basis::FullBasis) = last(basis.fullbasis)
secondlast(basis::FullBasis) = secondlast(basis.fullbasis)
all(basis::FullBasis) = all(basis.fullbasis)
all_withoutlasttwo(basis::FullBasis) = all_withoutlasttwo(basis.fullbasis)


function createbasis(maxdim::Int, x0::AbstractVecOrMat{E}, rot::ReOrthTechnique, basistype::Type{<:OrthonormalBasis}, pu::ProcessingUnit) where {E<:Real}
    if isa(rot,FullRO)
        fullbasis = basistype{E}(maxdim, x0, pu)
        return FullBasis(fullbasis)
    elseif isa(rot,PartialRO)
        throw(error("Finsih this for PartialRO"))
        # fullbasis = basistype{T}(maxdim, hilbertspacedim, s, pu)
        # partialbasis = basistype{T}(smalldim, hilbertspacedim, s, pu)
        # return PartialBasis(fullbasis, partialbasis)
    else
        throw(ArgumentError("Invalid reorthogonalization type. Use :full or :partial."))
    end
end


