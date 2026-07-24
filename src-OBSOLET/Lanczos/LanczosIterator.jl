
"""
    LanczosIterator(f!::Function, x0::AbstractVecOrMat, rot::ReOrthTechnique)

Container for immutable Lanczos iteration inputs:
- operator callback `f!(Y, X)`,
- initial seed vector/block `x0`,
- reorthogonalization strategy `rot`.
"""
struct LanczosIterator{T<:AbstractVecOrMat, ROT<:ReOrthTechnique}
    f!::Function
    x0::T
    rot::ROT
    function LanczosIterator(
        f!::Function,
        x0::T,
        rot::ROT
    ) where {T<:AbstractVecOrMat, ROT<:ReOrthTechnique}
        eltype(x0) <: Number || throw(ArgumentError("Lanczos requires numeric vector elements."))
        return new{T,ROT}(f!, x0, rot)
    end
end
