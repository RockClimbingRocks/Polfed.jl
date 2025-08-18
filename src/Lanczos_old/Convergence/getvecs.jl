


function calculate_eigenvectors!(convergenceinfo::ConvergenceInfo, factorization::KrylovFactorization, vecs::AbstractMatrix, ϕ::AbstractMatrix, idxs::AbstractVector)
    krylovbasis = all(factorization.basis)

    converged = convergenceinfo.converged

    idxs_ = view(idxs, 1:converged)
    # ϕ_howmany = view(ϕ, :, idxs_)
    ϕ_howmany = ϕ[:, idxs_]
    vecs_howmany = view(vecs, :, 1:converged)

    calculate_eigenvectors!_(vecs_howmany, krylovbasis, ϕ_howmany)
end





function calculate_eigenvectors!_(vecs::AbstractMatrix, krylovbasis::AbstractMatrix, ϕ::AbstractMatrix)
    mul!(vecs, krylovbasis, ϕ)
end



function calculate_eigenvectors!_(vecs::AbstractMatrix, krylovbasis::Vector{<:AbstractVector}, ϕ::AbstractMatrix)
    howmany = size(ϕ,2)
    krylovdim = length(krylovbasis)
    for i in 1:howmany
        veci = view(vecs,:, i)
        veci .= 0
        CUDA.@allowscalar for j in 1:krylovdim
            veci .+= krylovbasis[j] * ϕ[j, i]
        end
    end
end



function calculate_eigenvectors!_(vecs::AbstractMatrix, krylovbasis::Vector{<:AbstractMatrix}, ϕ::AbstractMatrix)
    howmany = size(ϕ,2)
    krylovdim = size(ϕ,1)
    # numofkrylovvecs = length(krylovbasis)
    s = size(krylovbasis[1],2)
    @inbounds for i in 1:howmany
        vecs[:, i] .= 0
        for j in 1:s:krylovdim
            l = (j+s-1)÷s
            ϕ_ = @view(ϕ[j:j+s-1, i])
            vecs[:, i] .+= krylovbasis[l] * ϕ_
        end
    end
end
