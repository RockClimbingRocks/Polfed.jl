"""
    calc_eigen_residuals(factorization::KrylovFactorization, ϕ::AbstractMatrix, idxs, mapvals)
        -> (vals, vecs, residuals)

Reconstruct the eigenvectors in the original space for the Ritz indices `idxs`, evaluate
their eigenvalues as Rayleigh quotients under the original operator `mapvals`, and return
the **true eigen-residuals** `‖A·vᵢ − λᵢ·vᵢ‖` (aligned with `idxs`).

The eigenvectors are stored per the factorization's layout: columns (`dim × nconv`) for
[`ColumnLayout`](@ref), rows (`nconv × dim`) for [`RowLayout`](@ref). `mapvals` is applied
to each eigenvector in that same orientation.

This is the *expensive* convergence signal — it reconstructs full-space eigenvectors from
the Krylov basis — so [`check_convergence!`](@ref) only calls it once the cheap Ritz
residuals ([`calc_ritz_residuals`](@ref)) already indicate the target is converged (or on
the final iteration).
"""
function calc_eigen_residuals(factorization::KrylovFactorization, ϕ::AbstractMatrix, idxs, mapvals)
    pu = factorization.pu
    layout = factorization.layout
    vecs = build_eigenvectors(factorization, ϕ, idxs)
    vals = build_eigenvalues(layout, vecs, mapvals, pu)
    residuals = build_residuals(layout, vecs, vals, mapvals, pu)
    return vals, vecs, residuals
end

# --- per-eigenvector access by layout --------------------------------------------------
# Column: eigenvectors are columns (dim × nconv); Row: eigenvectors are rows (nconv × dim).

@inline _neigvecs(::ColumnLayout, vecs::AbstractMatrix) = size(vecs, 2)
@inline _neigvecs(::RowLayout,    vecs::AbstractMatrix) = size(vecs, 1)

@inline _eigvec(::ColumnLayout, vecs::AbstractMatrix, i::Int) = view(vecs, :, i)
@inline _eigvec(::RowLayout,    vecs::AbstractMatrix, i::Int) = view(vecs, i:i, :)

@inline _eigvec_workspace(::ColumnLayout, vecs::AbstractMatrix, pu::ProcessingUnit) =
    pu.Vector{eltype(vecs)}(undef, size(vecs, 1))
@inline _eigvec_workspace(::RowLayout, vecs::AbstractMatrix, pu::ProcessingUnit) =
    pu.Matrix{eltype(vecs)}(undef, 1, size(vecs, 2))

# --- eigenvector reconstruction --------------------------------------------------------

"""
    build_eigenvectors(factorization::KrylovFactorization, ϕ::AbstractMatrix, idxs::AbstractVector) -> AbstractMatrix

Allocate and reconstruct the eigenvectors in the original space for the selected Ritz
indices `idxs`, from the Krylov basis and the Ritz vectors `ϕ[:, idxs]`. Column layout
returns a `dim × length(idxs)` matrix (eigenvectors as columns); row layout a
`length(idxs) × dim` matrix (eigenvectors as rows).
"""
function build_eigenvectors(factorization::KrylovFactorization, ϕ::AbstractMatrix, idxs::AbstractVector)
    pu = factorization.pu
    layout = factorization.layout
    T = eltype(factorization.r)
    dim = opdim(layout, factorization.r)
    nconv = length(idxs)

    vecs = layout isa RowLayout ? pu.Matrix{T}(undef, nconv, dim) : pu.Matrix{T}(undef, dim, nconv)
    nconv == 0 && return vecs

    krylovbasis = allvecs(factorization.basis)
    ϕ_sel = ϕ[:, idxs]
    _reconstruct_eigenvectors!(layout, vecs, krylovbasis, ϕ_sel)
    return vecs
end

"""
    _reconstruct_eigenvectors!(layout, vecs, krylovbasis, ϕ) -> nothing

Backend-specific eigenvector reconstruction for the different basis representations
(single matrix column/row, vector-of-vectors, vector-of-blocks).
"""
_reconstruct_eigenvectors!(::ColumnLayout, vecs::AbstractMatrix, krylovbasis::AbstractMatrix, ϕ::AbstractMatrix) =
    mul!(vecs, krylovbasis, ϕ)                 # dim×nconv = (dim×k)(k×nconv)
_reconstruct_eigenvectors!(::RowLayout, vecs::AbstractMatrix, krylovbasis::AbstractMatrix, ϕ::AbstractMatrix) =
    mul!(vecs, ϕ', krylovbasis)                # nconv×dim = (nconv×k)(k×dim)

"""Vector-of-vectors basis overload (column semantics — `VectorBasis` is column-only)."""
function _reconstruct_eigenvectors!(::Layout, vecs::AbstractMatrix, krylovbasis::Vector{<:AbstractVector}, ϕ::AbstractMatrix)
    howmany = size(ϕ, 2)
    krylovdim = length(krylovbasis)
    for i in 1:howmany
        veci = view(vecs, :, i)
        veci .= 0
        gpu_allowscalar() do
            for j in 1:krylovdim
                veci .+= krylovbasis[j] * ϕ[j, i]
            end
        end
    end
end

"""Vector-of-blocks basis overload (column semantics — `VectorBasis` is column-only)."""
function _reconstruct_eigenvectors!(::Layout, vecs::AbstractMatrix, krylovbasis::Vector{<:AbstractMatrix}, ϕ::AbstractMatrix)
    howmany = size(ϕ, 2)
    krylovdim = size(ϕ, 1)
    s = size(krylovbasis[1], 2)
    @inbounds for i in 1:howmany
        vecs[:, i] .= 0
        for j in 1:s:krylovdim
            l = (j+s-1) ÷ s
            ϕ_ = @view(ϕ[j:j+s-1, i])
            vecs[:, i] .+= krylovbasis[l] * ϕ_
        end
    end
end

# --- eigenvalues and residuals ---------------------------------------------------------

"""
    build_eigenvalues(layout, vecs::AbstractMatrix, valsmap!, pu::ProcessingUnit) -> AbstractVector

Eigenvalues of the reconstructed eigenvectors as Rayleigh quotients `λᵢ = ⟨vᵢ | A | vᵢ⟩`
under the original operator `valsmap!`, applied per `layout`. Returns a real vector on `pu`.
"""
function build_eigenvalues(layout::Layout, vecs::AbstractMatrix, valsmap!, pu::ProcessingUnit)
    S = real(eltype(vecs))
    nconv = _neigvecs(layout, vecs)
    vals_cpu = zeros(S, nconv)
    nconv == 0 && return pu.Vector{S}(vals_cpu)

    Hv = _eigvec_workspace(layout, vecs, pu)
    for i in 1:nconv
        v = _eigvec(layout, vecs, i)
        valsmap!(Hv, v)
        vals_cpu[i] = real(dot(vec(v), vec(Hv)))
    end
    return pu.Vector{S}(vals_cpu)
end

"""
    build_residuals(layout, vecs::AbstractMatrix, vals::AbstractVector, valsmap!, pu::ProcessingUnit) -> Vector{Float64}

True eigen-residuals `‖A·vᵢ − λᵢ·vᵢ‖` for each extracted eigenpair under `valsmap!`,
aligned with the eigenvectors of `vecs` (per `layout`).
"""
function build_residuals(layout::Layout, vecs::AbstractMatrix, vals::AbstractVector, valsmap!, pu::ProcessingUnit)
    nconv = _neigvecs(layout, vecs)
    res = zeros(Float64, nconv)
    nconv == 0 && return res

    r = _eigvec_workspace(layout, vecs, pu)
    vals_cpu = Vector(vals)
    for i in 1:nconv
        v = _eigvec(layout, vecs, i)
        valsmap!(r, v)
        r .-= vals_cpu[i] .* v
        res[i] = norm(vec(r))
    end
    return res
end
