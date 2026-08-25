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
    vals, residuals = build_values_and_residuals(layout, vecs, mapvals, pu)
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
        fill!(veci, zero(eltype(veci)))
        gpu_allowscalar() do
            for j in 1:krylovdim
                axpy!(ϕ[j, i], krylovbasis[j], veci)
            end
        end
    end
    return nothing
end

"""Vector-of-blocks basis overload (column semantics — `VectorBasis` is column-only)."""
function _reconstruct_eigenvectors!(::Layout, vecs::AbstractMatrix, krylovbasis::Vector{<:AbstractMatrix}, ϕ::AbstractMatrix)
    howmany = size(ϕ, 2)
    krylovdim = size(ϕ, 1)
    s = size(krylovbasis[1], 2)
    @inbounds for i in 1:howmany
        veci = view(vecs, :, i)
        fill!(veci, zero(eltype(veci)))
        for j in 1:s:krylovdim
            l = (j+s-1) ÷ s
            ϕ_ = @view(ϕ[j:j+s-1, i])
            mul!(veci, krylovbasis[l], ϕ_, one(eltype(veci)), one(eltype(veci)))
        end
    end
    return nothing
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
        axpy!(-vals_cpu[i], vec(v), vec(r))
        res[i] = norm(vec(r))
    end
    return res
end

"""
    build_values_and_residuals(layout, vecs::AbstractMatrix, valsmap!, pu::ProcessingUnit)
        -> (vals, residuals)

Rayleigh quotients and true eigen-residuals in **one pass**: `λᵢ = ⟨vᵢ|A|vᵢ⟩` and
`‖A·vᵢ − λᵢ·vᵢ‖`, both read off the same `A·vᵢ`.

Equivalent to [`build_eigenvalues`](@ref) followed by [`build_residuals`](@ref) — which are
kept as standalone helpers — but applies `valsmap!` **once** per eigenvector instead of
twice. The residual only needs `A·vᵢ` shifted by `λᵢ·vᵢ`, and `λᵢ` is known the moment
`A·vᵢ` is: recomputing it was half the map-back cost of every convergence check.

`valsmap!` is applied to a single eigenvector at a time, in the layout's orientation, so the
callback contract is unchanged.
"""
function build_values_and_residuals(layout::Layout, vecs::AbstractMatrix, valsmap!,
                                    pu::ProcessingUnit)
    S = real(eltype(vecs))
    nconv = _neigvecs(layout, vecs)
    vals_cpu = zeros(S, nconv)
    res = zeros(Float64, nconv)
    nconv == 0 && return pu.Vector{S}(vals_cpu), res

    w = _eigvec_workspace(layout, vecs, pu)
    for i in 1:nconv
        v = _eigvec(layout, vecs, i)
        valsmap!(w, v)                          # the one operator application
        λ = real(dot(vec(v), vec(w)))
        vals_cpu[i] = λ
        axpy!(-λ, vec(v), vec(w))               # w ← A·v − λ·v, in place
        res[i] = norm(vec(w))
    end
    return pu.Vector{S}(vals_cpu), res
end

# --- reuse of a reconstruction across one convergence check -----------------------------

"""
    _EigenBatch

One [`calc_eigen_residuals`](@ref) result, tagged with the Ritz indices it was built for.

[`check_convergence!`](@ref) records every batch a check reconstructs so the terminal
extraction can be *sliced* out of them rather than reconstructed again — see
[`_reuse_eigen_batches`](@ref).
"""
struct _EigenBatch{V,M,R}
    idxs::Vector{Int}
    vals::V
    vecs::M
    res::R
end

@inline _select_eigvecs(::ColumnLayout, vecs::AbstractMatrix, pos::AbstractVector{Int}) = vecs[:, pos]
@inline _select_eigvecs(::RowLayout,    vecs::AbstractMatrix, pos::AbstractVector{Int}) = vecs[pos, :]

@inline _concat_eigvecs(::ColumnLayout, parts::AbstractVector) = reduce(hcat, parts)
@inline _concat_eigvecs(::RowLayout,    parts::AbstractVector) = reduce(vcat, parts)

"""
    _reuse_eigen_batches(layout, batches, keep) -> Union{Nothing, Tuple}

Serve the eigenpairs for the Ritz indices `keep` out of already-reconstructed `batches`,
or `nothing` when they cannot be (some index was never reconstructed this check, or the
batches interleave in `keep`) and the caller must rebuild.

Reconstruction is a `nconv × krylovdim × dim` GEMM — the dominant cost of a convergence
check — and the terminal extraction always asks for a *subset* of what deciding the target
already built. `keep` is a concatenation of one contiguous stretch per spectral end, so each
batch contributes one slice; when a single batch covers `keep` in order, its arrays are
handed back with no copy at all.
"""
function _reuse_eigen_batches(layout::Layout, batches::AbstractVector, keep::AbstractVector{Int})
    (isempty(batches) || isempty(keep)) && return nothing

    # Locate every kept index: (which batch, which position inside it).
    ids  = Vector{Int}(undef, length(keep))
    poss = Vector{Int}(undef, length(keep))
    for (k, idx) in pairs(keep)
        found = false
        for (bi, b) in pairs(batches)
            pos = findfirst(==(idx), b.idxs)
            pos === nothing && continue
            ids[k], poss[k] = bi, pos
            found = true
            break
        end
        found || return nothing
    end

    # Each batch must occupy one contiguous stretch of `keep`, or a per-batch slice would
    # not reproduce the requested order.
    groups = unique(ids)
    ranges = Vector{UnitRange{Int}}(undef, length(groups))
    for (g, bi) in pairs(groups)
        first_k = findfirst(==(bi), ids)
        last_k  = findlast(==(bi), ids)
        all(==(bi), view(ids, first_k:last_k)) || return nothing
        ranges[g] = first_k:last_k
    end

    if length(groups) == 1
        b = batches[groups[1]]
        poss == eachindex(b.idxs) && return (b.vals, b.vecs, b.res)   # exact hit, no copy
        return (b.vals[poss], _select_eigvecs(layout, b.vecs, poss), b.res[poss])
    end

    vals = reduce(vcat, [batches[bi].vals[poss[r]] for (bi, r) in zip(groups, ranges)])
    vecs = _concat_eigvecs(layout,
                           [_select_eigvecs(layout, batches[bi].vecs, poss[r])
                            for (bi, r) in zip(groups, ranges)])
    res  = reduce(vcat, [batches[bi].res[poss[r]] for (bi, r) in zip(groups, ranges)])
    return (vals, vecs, res)
end
