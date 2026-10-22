"""
    LanczosIterator(f!, x0::AbstractVecOrMat, rot::ReOrthTechnique, layout::Layout=ColumnLayout())

Container for the immutable Lanczos iteration inputs:
- operator callback `f!(Y, X)` (writes `A·X` into `Y`),
- initial seed vector/block `x0` (its block dimension per `layout` sets the block size),
- reorthogonalization strategy `rot`,
- block storage orientation `layout` ([`ColumnLayout`](@ref) / [`RowLayout`](@ref)).

The callback type `F` is a type parameter so calls through `f!` specialize.
"""
struct LanczosIterator{F, T<:AbstractVecOrMat, ROT<:ReOrthTechnique, L<:Layout}
    f!::F
    x0::T
    rot::ROT
    layout::L

    function LanczosIterator(f!::F, x0::T, rot::ROT, layout::L=ColumnLayout()) where {F, T<:AbstractVecOrMat, ROT<:ReOrthTechnique, L<:Layout}
        eltype(x0) <: Number || throw(ArgumentError("Lanczos requires numeric vector elements."))
        return new{F,T,ROT,L}(f!, x0, rot, layout)
    end
end
