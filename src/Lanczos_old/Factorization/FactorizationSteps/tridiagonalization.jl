
# function tridiagonalization!(factorization::KrylovFactorization)

#     β = calc_norm_krylovvec!(factorization)

#     addnorm!(factorization, β)
#     add!(krylovbasis, factorization.r)
# end



# @inline function tridiagonalization!(factorization::LanczosFactorization)

#     factorization.krylovdim += 1
#     tridiagonalization!(factorization.βs, factorization.basis, factorization.r, factorization.krylovdim)
# end

# @inline function tridiagonalization!(factorization::BlockLanczosFactorization)

#     factorization.krylovdim += factorization.blockdim
#     tridiagonalization!(factorization.mat, factorization.basis, factorization.r, factorization.krylovdim, factorization.blockdim, factorization.pu)

# end


# @inline function tridiagonalization!(mat:: AbstractMatrix{T}, krylovbasis::Basis, W::AbstractMatrix{T}, k::Int64, s::Int64, pu::CPU) where {T <: Number}
#     QR = qr(W)

#     # mat[s*(k-1)+1 :  s*k    , s*(k-2)+1 : s*(k-1)  ] .= QR.R
#     # mat[s*(k-2)+1 : s*(k-1) , s*(k-1)+1 :  s*k     ] .= QR.R'


#     mat[k-s+1:k   , k-2s+1:k-s] .= QR.R
#     mat[k-2s+1:k-s, k-s+1:s] .= QR.R'

#     add!(krylovbasis, pu.mat(QR.Q))
# end


# @inline function tridiagonalization!(βs::AbstractVector, basis::Basis, W::AbstractVector, k::Int)
#     β = LinearAlgebra.norm(W)
#     βs[k-1] = β

#     # broadcast!(*, W, W, 1/β)
#     W ./= β
#     add!(basis, W)
# end
