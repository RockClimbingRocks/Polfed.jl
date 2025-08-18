
@inline function reorthogonalization!(factorization::KrylovFactorization, ::FullRO)
    krylovbasis = all_withoutlasttwo(factorization.basis)
    # krylovbasis = all(factorization.basis)

    reorthogonalization!(krylovbasis, factorization.r, factorization.pu)
end

@inline function reorthogonalization!(factorization::KrylovFactorization, ::PartialRO)
    throw(error("reorthogonalization! method not yet defined for PartialRO."))
end


############################################################################################################################
#                                       MATRIX GRAM-SCHMIDT REORTHOGONALIZATION 
############################################################################################################################

@inline function reorthogonalization!(krylovbasis::Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}}, W::AbstractVecOrMat{T}, _::ProcessingUnit) where {T<:Number}
    B_gpu, B_cpu = krylovbasis

    # Project out GPU part (W is on GPU)
    W .-= B_gpu * (B_gpu' * W)

    # Move W to CPU to project out CPU part
    W_cpu = Array(W)   # copy to CPU
    W_cpu .-= B_cpu * (B_cpu' * W_cpu)

    # Move back to GPU
    W .= CuArray(W_cpu)
end

@inline function reorthogonalization!(krylovbasis::AbstractMatrix{T}, W::AbstractVecOrMat{T}, _::ProcessingUnit) where {T<:Number}
    W .-= krylovbasis * (krylovbasis' * W)
end

 
############################################################################################################################
#                                       CLASSICAL GRAM-SCHMIDT REORTHOGONALIZATION 
############################################################################################################################

# @inline function reorthogonalization!(krylovbasis::AbstractMatrix{E}, W::AbstractVector{E}, _::ProcessingUni) where {E<:Number}
#     for i in 1:size(krylovbasis,2)
#         v = view(krylovbasis, :, i)
#         β = v'*W
#         W .-= v*β
#     end
# end

# @inline function reorthogonalization!(krylovbasis::AbstractMatrix{E}, W::AbstractMatrix{E}, pu::ProcessingUnit, ::ClassicalGramSchmidt) where {E<:Number}
#     s = size(W,2)
#     β = pu.Matrix{E}(undef, s, s)
#     for i in 1:s:size(krylovbasis,2)
#         v = view(krylovbasis, :, i:i+s-1)
#         mul!(β, v', W)
#         W .-= v*β
#     end
# end

# @inline function reorthogonalization!(krylovbasis::Vector{<:T}, W::T, _::ProcessingUnit, ::ClassicalGramSchmidt) where {T<:AbstractVector}
#     for v in krylovbasis    
#         β = v' * W
#         W .-= v*β
#     end
# end

# @inline function reorthogonalization!(krylovbasis::Vector{<:T}, W::T, pu::ProcessingUnit, ::ClassicalGramSchmidt) where {T<:AbstractMatrix}
#     s = size(W,2)
#     β =  pu.Matrix{eltype(W)}(undef, s, s)

#     for v in krylovbasis    
#         mul!(β, v', W)
#         W .-= v*β
#     end
# end
