
struct LanczosIterator{F<:Function, T<:AbstractVecOrMat, ROT<:ReOrthTechnique, RO<:ReOrthogonalizer}
    f!::F
    x₀::T
    rot::ROT
    reorth::RO
    function LanczosIterator(f!::F,
                             x₀::T,
                             rot::ROT,
                             reorth::RO) where {F<:Function, T<:AbstractVecOrMat, ROT<:ReOrthTechnique, RO<:ReOrthogonalizer}

        return new{F,T,ROT, RO}(f!, x₀, rot, reorth)
    end
end

