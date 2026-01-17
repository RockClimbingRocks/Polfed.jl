
@inline function reorthogonalization!(factorization::KrylovFactorization, ::FullRO)
    krylovbasis = all_withoutlasttwo(factorization.basis)
    # krylovbasis = all(factorization.basis)


    # display(krylovbasis)

    # println()
    # println()
    # println()

    # display(factorization.r)

    # println("Performing full reorthogonalization... (krylov basis dim = ", size(krylovbasis,2), ")")
    size(krylovbasis, 2) == 0 && (return)
    reorthogonalization!(krylovbasis, factorization.r, factorization.pu)
end



@inline function reorthogonalization!(factorization::KrylovFactorization, pro::PartialRO)
    # throw(error("reorthogonalization! method not yet defined for PartialRO."))
    krylovdim = factorization.krylovdim
    blocksize =  factorization isa LanczosFactorization ? 1 : factorization.blocksize
    kryloviter = Integer(krylovdim ÷ blocksize)
    pro_condition = (kryloviter%(2+pro.skip)==0) || ((kryloviter-1)%(2+pro.skip)==0)



    if pro_condition && krylovdim > 2*blocksize
        # println("Performing reorthogonalization... (at iter ", kryloviter, ")")
        krylovbasis = all_withoutlasttwo(factorization.basis)
        reorthogonalization!(krylovbasis, factorization.r, factorization.pu)
    end

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
    # println("Im innnnn....")
    # nvecs = size(krylovbasis,2)
    # iszero(nvecs) && (println("Skipping reorthogonalization (because nvecs == 0)..."); return nothing)
    # println("doing reorthogonalization")

    W .-= krylovbasis * (krylovbasis' * W)
end

 

############################################################################################################################
#                                       CLASSICAL GRAM-SCHMIDT REORTHOGONALIZATION 
############################################################################################################################

# @inline function reorthogonalization!(krylovbasis::AbstractMatrix{E}, W::AbstractVector{E}, _::ProcessingUnit) where {E<:Number}
#     for i in 1:size(krylovbasis,2)
#         v = view(krylovbasis, :, i)
#         β = v'*W
#         W .-= v*β
#     end
# end

# @inline function reorthogonalization!(krylovbasis::AbstractMatrix{E}, W::AbstractMatrix{E}, pu::ProcessingUnit) where {E<:Number}
#     s = size(W,2)
#     β = pu.Matrix{E}(undef, s, s)
#     for i in 1:s:size(krylovbasis,2)
#         v = view(krylovbasis, :, i:i+s-1)
#         mul!(β, v', W)
#         W .-= v*β
#     end
# end
