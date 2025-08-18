
function calculate_eigenresiduals!(
    convergenceinfo::ConvergenceInfo, 
    valsinfo::EigenvaluesInfo, 
    vecs::AbstractMatrix{E}, 
    vals::AbstractVector{E}, 
    pu::ProcessingUnit
) where {E<:Real}
    tol = valsinfo.tol

    r = pu.vec{E}(undef, size(vecs,1))

    vals_cpu = Vector{E}(vals)

    converged = 0
    do_count = true
    maxresidual = 0.

    for i in 1:convergenceinfo.converged
        vec = view(vecs, :, i)
        valsinfo.mapvals(r, vec)

        r .-= vals_cpu[i] .* vec

        residual = norm(r)

        (do_count && residual < tol) ? (converged += 1) : (do_count = false)
        residual > maxresidual && (maxresidual = residual)
    end

    convergenceinfo.converged > converged && (@info "Eigenvalues converged: $converged out of $(convergenceinfo.converged)")
    valsinfo.converged = converged
    valsinfo.eigenresidual = maxresidual
end

function calculate_residuals!(convergenceinfo::ConvergenceInfo, factorization::KrylovFactorization)
    mat = constructfactorizedmat(factorization)
    # display(mat)

    λ, ϕ = eigen(mat)
    idxs = sortvals(λ, convergenceinfo.sorter)
    β = getnorm(factorization)
    

    tol = convergenceinfo.tol
    converged = 0
    maxresidual = 0.

    for i in 1:convergenceinfo.howmany
        position_i = idxs[i]
        residual = calculateresidual(ϕ, β, position_i)

        # do not interchange these two lines! They do not commute! 
        residual > maxresidual && (maxresidual = residual)
        maxresidual < tol && (converged += 1)
    end
    

    # println("current residual: ", convergenceinfo.residual)

    convergenceinfo.numofchecks += 1 

    convergenceinfo.converged = converged
    convergenceinfo.residual = maxresidual
    push!(convergenceinfo.converged_history, converged)
    push!(convergenceinfo.krylovbasisdim_history, factorization.krylovdim)
    push!(convergenceinfo.maxresidual_history, maxresidual)


    return λ, ϕ, idxs
end

function calculateresidual(ϕ::AbstractMatrix, β::AbstractMatrix, i::Int)
    s = size(β,1)
    return norm(β*ϕ[end-s+1:end,i])
end

function calculateresidual(ϕ::AbstractMatrix, β::Real, i::Int)
    return CUDA.@allowscalar abs(β*ϕ[end,i])
end
