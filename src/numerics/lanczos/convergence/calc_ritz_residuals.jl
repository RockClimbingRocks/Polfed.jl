"""
    calc_ritz_residuals(factorization::KrylovFactorization) -> (λ, ϕ, residuals)

Diagonalize the projected Krylov matrix and return the ascending Ritz values `λ`, the
Ritz vectors `ϕ` (columns), and the **Ritz residual estimate for every Ritz pair**
(`residuals[i]` aligned with `λ[i]`).

The projected matrix is `SymTridiagonal` (scalar Lanczos) or `Hermitian` (block), so
`eigen` uses the fast symmetric solver. This is the *cheap* convergence signal: it uses
only the trailing components of `ϕ` coupled through the **current unnormalised Lanczos
residual** `factorization.r`, so it is computed on every scheduled check (unlike the eigen
residual — see [`calc_eigen_residuals`](@ref)).

The coupling stored in the projected matrix is `β_{m-1}`; using it here is an off-by-one
error. For the current `m`-dimensional projection the residual is controlled by `β_m`, which
is precisely the norm/factor still held in `factorization.r` before the next Lanczos step.
"""
function calc_ritz_residuals(factorization::KrylovFactorization)
    mat = constructfactorizedmat(factorization)
    λ, ϕ = eigen(mat)
    λcpu = is_gpu_array(λ) ? collect(λ) : λ
    residuals = _current_ritz_residuals(factorization, ϕ, eachindex(λcpu))
    return λcpu, ϕ, residuals
end

function _current_ritz_residuals(factorization::LanczosFactorization,
                                 ϕ::AbstractMatrix, idxs)
    β = norm(factorization.r)
    β < _breakdown_tol(eltype(factorization.r)) && (β = zero(β))
    return Float64[calculateresidual(ϕ, β, i) for i in idxs]
end

function _current_ritz_residuals(factorization::BlockLanczosFactorization,
                                 ϕ::AbstractMatrix, idxs)
    if norm(factorization.r) < _breakdown_tol(eltype(factorization.r))
        return zeros(Float64, length(idxs))
    end

    # Column layout: G = r' r. Row layout: G = r r'. In both cases the squared
    # residual of trailing Ritz block z is z'Gz.
    G = overlap(factorization.layout, factorization.r, factorization.r)
    s = factorization.blocksize
    return Float64[_gram_residual(ϕ, G, s, i) for i in idxs]
end

function _gram_residual(ϕ::AbstractMatrix, G::AbstractMatrix, s::Int, i::Int)
    z = view(ϕ, size(ϕ, 1)-s+1:size(ϕ, 1), i)
    squared = real(dot(z, G * z))
    return sqrt(max(zero(squared), squared))
end

"""
    calculateresidual(ϕ, β, i) -> Real

Ritz-residual estimate for Ritz vector `i`, from the trailing components of `ϕ`
coupled through a supplied scalar or block factor. These public helpers are retained for
source compatibility; [`calc_ritz_residuals`](@ref) obtains the current coupling directly
from the factorization.
"""
function calculateresidual(ϕ::AbstractMatrix, β::AbstractMatrix, i::Int)
    s = size(β, 1)
    return norm(β * ϕ[end-s+1:end, i])
end

function calculateresidual(ϕ::AbstractMatrix, β::Real, i::Int)
    return gpu_allowscalar() do
        abs(β * ϕ[end, i])
    end
end
