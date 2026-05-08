
"""
    calculate_eigenresiduals!(convergenceinfo::ConvergenceInfo, vecs::AbstractMatrix, vals::AbstractVector, pu::ProcessingUnit) -> nothing

Compute residual norms `||A*v - λ*v||` for extracted eigenpairs and update
eigen-convergence counters in `convergenceinfo`.
"""
function calculate_eigenresiduals!(
    convergenceinfo::ConvergenceInfo,
    vecs::AbstractMatrix,
    vals::AbstractVector,
    pu::ProcessingUnit
) 
    eigentol = convergenceinfo.eigentol

    r = pu.Vector{eltype(vecs)}(undef, size(vecs,1))

    vals_cpu = Vector{eltype(vals)}(vals)

    converged = 0
    do_count = true
    maxresidual = 0.

    for i in 1:convergenceinfo.converged
        vec = view(vecs, :, i)
        convergenceinfo.mapvals(r, vec)

        r .-= vals_cpu[i] .* vec

        residual = norm(r)

        (do_count && residual < eigentol) ? (converged += 1) : (do_count = false)
        residual > maxresidual && (maxresidual = residual)
    end

    convergenceinfo.converged > converged && polfed_log(
        POLFED_INFO_LEVEL,
        "Eigenvalues converged: $converged out of $(convergenceinfo.converged)",
    )
    convergenceinfo.eigenconverged  = converged
    convergenceinfo.eigenresidual   = maxresidual
end

"""
    calculate_residuals!(convergenceinfo::ConvergenceInfo, factorization::KrylovFactorization)

Compute Ritz spectrum/residuals from projected Krylov matrix and update
convergence statistics.

# Returns
- `(λ, ϕ, idxs)` where `λ` are Ritz values, `ϕ` Ritz vectors, and `idxs`
  sorting indices per current convergence policy.
"""
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

"""
    calculateresidual(ϕ, β, i) -> Real

Compute residual estimate for Ritz vector `i`.

Supports scalar and block Lanczos coupling (`β::Real` or `β::AbstractMatrix`).
"""
function calculateresidual(ϕ::AbstractMatrix, β::AbstractMatrix, i::Int)
    s = size(β,1)
    return norm(β*ϕ[end-s+1:end,i])
end

function calculateresidual(ϕ::AbstractMatrix, β::Real, i::Int)
    return gpu_allowscalar() do
        abs(β*ϕ[end,i])
    end
end
