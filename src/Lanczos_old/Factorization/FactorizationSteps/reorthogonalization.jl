
@inline function reorthogonalization!(factorization::KrylovFactorization, ::FullRO, reorth::ReOrthogonalizer)
    krylovbasis = all_withoutlasttwo(factorization.basis)
    # krylovbasis = all(factorization.basis)

    reorthogonalization!(krylovbasis, factorization.r, factorization.pu, reorth)
end

@inline function reorthogonalization!(factorization::KrylovFactorization, ::PartialRO, reorth::ReOrthogonalizer)
    throw(error("reorthogonalization! method not yet defined for PartialRO."))
end


############################################################################################################################
#                                       MATRIX GRAM-SCHMIDT REORTHOGONALIZATION 
############################################################################################################################

@inline function reorthogonalization!(krylovbasis::AbstractMatrix{T}, W::AbstractVecOrMat{T}, _::ProcessingUnit, ::MatrixGramSchmidt) where {T<:Number}
    W .-= krylovbasis * (krylovbasis' * W)
end

 
############################################################################################################################
#                                       CLASSICAL GRAM-SCHMIDT REORTHOGONALIZATION 
############################################################################################################################

@inline function reorthogonalization!(krylovbasis::AbstractMatrix{E}, W::AbstractVector{E}, _::ProcessingUnit, ::ClassicalGramSchmidt) where {E<:Number}
    for i in 1:size(krylovbasis,2)
        v = view(krylovbasis, :, i)
        β = v'*W
        W .-= v*β
    end
end

@inline function reorthogonalization!(krylovbasis::AbstractMatrix{E}, W::AbstractMatrix{E}, pu::ProcessingUnit, ::ClassicalGramSchmidt) where {E<:Number}
    s = size(W,2)
    β = pu.mat{E}(undef, s, s)
    for i in 1:s:size(krylovbasis,2)
        v = view(krylovbasis, :, i:i+s-1)
        mul!(β, v', W)
        W .-= v*β
    end
end

@inline function reorthogonalization!(krylovbasis::Vector{<:T}, W::T, _::ProcessingUnit, ::ClassicalGramSchmidt) where {T<:AbstractVector}
    for v in krylovbasis    
        β = v' * W
        W .-= v*β
    end
end

@inline function reorthogonalization!(krylovbasis::Vector{<:T}, W::T, pu::ProcessingUnit, ::ClassicalGramSchmidt) where {T<:AbstractMatrix}
    s = size(W,2)
    β =  pu.mat{eltype(W)}(undef, s, s)

    for v in krylovbasis    
        mul!(β, v', W)
        W .-= v*β
    end
end
