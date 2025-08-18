


function calculate_eigenvectors!(convergenceinfo::ConvergenceInfo, factorization::KrylovFactorization, vecs::AbstractMatrix, ϕ::AbstractMatrix, idxs::AbstractVector)
    krylovbasis = all(factorization.basis)

    converged = convergenceinfo.converged

    idxs_ = view(idxs, 1:converged)
    # ϕ_howmany = view(ϕ, :, idxs_)
    ϕ_howmany = ϕ[:, idxs_]
    vecs_howmany = view(vecs, :, 1:converged)

    calculate_eigenvectors!_(vecs_howmany, krylovbasis, ϕ_howmany)
end



function calculate_eigenvectors!_(vecs::AbstractMatrix, krylovbasis::Tuple{AM_gpu, AM_cpu}, ϕ::AbstractMatrix) where {AM_gpu<:AbstractMatrix, AM_cpu<:AbstractMatrix}
    B_gpu, B_cpu = krylovbasis
    nvecs_gpu = size(B_gpu, 2)
    nvecs_cpu = size(B_cpu, 2)

    # Start with GPU part
    ϕ_gpu = view(ϕ, 1:nvecs_gpu, :)
    mul!(vecs, B_gpu, ϕ_gpu)

    # Add CPU part (move to CPU first)
    vecs_cpu = zeros(eltype(vecs), size(vecs))
    ϕ_cpu = Matrix(view(ϕ, nvecs_gpu+1:nvecs_gpu+nvecs_cpu, :))
    vecs_cpu .= B_cpu * ϕ_cpu

    vecs .+= CuMatrix(vecs_cpu)
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
