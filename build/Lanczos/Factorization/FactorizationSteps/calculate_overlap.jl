
"""
    calculate_overlap!(factorization::KrylovFactorization) -> nothing

Compute and store overlap/projection coefficients for the current iteration.
"""
@inline function calculate_overlap!(factorization::KrylovFactorization)
    calcoverlap!(factorization)
end




# @inline function calculate_overlap!(factorization::LanczosFactorization)
#     vec = factorization.v_last
#     Avec = factorization.r
    
#     inner!(factorization.αs, vec, Avec, factorization.krylovdim) 
# end

# @inline function inner!(αs::AbstractVector{E}, vec::AbstractVecOrMat{E}, Avec::AbstractVector{E}, k::Int) where {E<:Real}
#     αs[k] = dot(vec, Avec)
# end


# @inline function inner!(mat::AbstractMatrix{E}, vecs::AbstractMatrix{E}, Avecs::AbstractMatrix{E}, k::Int, s::Int) where {E<:Real}
#     α_k = view(mat, k-s+1:k, k-s+1:k);

#     mul!(α_k, vecs', Avecs);
#     α_k .= Symmetric(α_k); # mat[s*(k-1)+1 : s*k,  s*(k-1)+1 : s*k] .= Symmetric(α_k) 
# end
