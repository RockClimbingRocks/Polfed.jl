
struct LanczosIterator{E<:Real, T<:AbstractVecOrMat{E}, ROT<:ReOrthTechnique}
    f!::Function
    x0::T
    rot::ROT
    function LanczosIterator(
        f!::Function,
        x0::T,
        rot::ROT
    ) where {E<:Real, T<:AbstractVecOrMat{E}, ROT<:ReOrthTechnique}

        return new{E,T,ROT}(f!, x0, rot)
    end
end

