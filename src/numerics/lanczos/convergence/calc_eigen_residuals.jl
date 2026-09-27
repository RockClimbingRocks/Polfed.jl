"""
    calc_eigen_residuals(factorization::KrylovFactorization, ϕ::AbstractMatrix, idxs, mapvals)
        -> (vals, vecs, residuals)

Reconstruct the eigenvectors in the original space for the Ritz indices `idxs`, evaluate
their eigenvalues as Rayleigh quotients under the original operator `mapvals`, and return
the **true eigen-residuals** `‖A·vᵢ − λᵢ·vᵢ‖` (aligned with `idxs`).

The eigenvectors are stored per the factorization's layout: columns (`dim × nconv`) for
[`ColumnLayout`](@ref), rows (`nconv × dim`) for [`RowLayout`](@ref). `mapvals` is applied
to each eigenvector in that same orientation.

Allocates a full eigenvector matrix next to the Krylov basis. [`check_convergence!`](@ref)
therefore uses [`calc_eigen_residuals_chunked`](@ref) to decide and
[`extract_eigenpairs!`](@ref) to extract; this form is their reference and the extraction's
fallback for a non-matrix basis.
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

    _values_and_residuals!(vals_cpu, res, layout, vecs, valsmap!, _eigvec_workspace(layout, vecs, pu))
    return pu.Vector{S}(vals_cpu), res
end

"""
    _values_and_residuals!(vals, res, layout, vecs, valsmap!, w) -> nothing

The loop of [`build_values_and_residuals`](@ref), writing into CPU `vals`/`res` and using the
caller's single-eigenvector workspace `w`.
"""
function _values_and_residuals!(vals::AbstractVector, res::AbstractVector, layout::Layout,
                                vecs::AbstractMatrix, valsmap!, w::AbstractArray)
    for i in 1:_neigvecs(layout, vecs)
        v = _eigvec(layout, vecs, i)
        valsmap!(w, v)                          # the one operator application
        λ = real(dot(vec(v), vec(w)))
        vals[i] = λ
        axpy!(-λ, vec(v), vec(w))               # w ← A·v − λ·v, in place
        res[i] = norm(vec(w))
    end
    return nothing
end

# --- memory-bounded variants: the Krylov basis is the only full-size array --------------

"""Eigenvectors rebuilt per pass by [`calc_eigen_residuals_chunked`](@ref)."""
const _EIGEN_CHUNK = 64

"""Byte budget of the scratch block used by the in-place extraction and permutation."""
const _SCRATCH_BYTES = 64 * 2^20

# Scratch-block width for `n` eigenvectors along a length-`dim` axis: within `_SCRATCH_BYTES`,
# but at least 256 so the block GEMM stays efficient.
_scratch_width(n::Int, dim::Int, ::Type{T}) where {T} =
    clamp(fld(_SCRATCH_BYTES, max(n, 1) * sizeof(T)), min(256, dim), dim)

# The leading `n` eigenvector slots of a layout-oriented matrix.
@inline _leading(::RowLayout,    M::AbstractMatrix, n::Int) = view(M, 1:n, :)
@inline _leading(::ColumnLayout, M::AbstractMatrix, n::Int) = view(M, :, 1:n)

"""
    calc_eigen_residuals_chunked(factorization, ϕ, idxs, mapvals; chunk=64) -> (vals, residuals)

[`calc_eigen_residuals`](@ref) without keeping the eigenvectors: they are rebuilt `chunk` at a
time in one reusable buffer, so the workspace is `chunk` vectors however many pairs are
checked. Leaves the Krylov basis intact. `vals` is a CPU vector.
"""
function calc_eigen_residuals_chunked(factorization::KrylovFactorization, ϕ::AbstractMatrix,
                                      idxs::AbstractVector, mapvals; chunk::Int=_EIGEN_CHUNK)
    pu, layout = factorization.pu, factorization.layout
    T = eltype(factorization.r)
    n = length(idxs)
    vals, res = Vector{real(T)}(undef, n), Vector{Float64}(undef, n)
    n == 0 && return vals, res

    dim = opdim(layout, factorization.r)
    c = min(chunk, n)
    buf = layout isa RowLayout ? pu.Matrix{T}(undef, c, dim) : pu.Matrix{T}(undef, dim, c)
    w = _eigvec_workspace(layout, buf, pu)
    krylovbasis = allvecs(factorization.basis)
    for lo in 1:c:n
        hi = min(lo + c - 1, n)
        vecs = _leading(layout, buf, hi - lo + 1)
        _reconstruct_eigenvectors!(layout, vecs, krylovbasis, ϕ[:, idxs[lo:hi]])
        _values_and_residuals!(view(vals, lo:hi), view(res, lo:hi), layout, vecs, mapvals, w)
    end
    return vals, res
end

"""
    extract_eigenpairs!(factorization, ϕ, idxs, mapvals; block=0) -> (vals, vecs, residuals)

The final extraction. Same result as [`calc_eigen_residuals`](@ref), but for a
[`MatrixBasis`](@ref) the eigenvectors are built inside the basis storage — its leading
`length(idxs)` rows (row layout) or columns (column layout) — and `vecs` is a view of them, so
no eigenvector matrix is allocated. Other bases fall back to [`calc_eigen_residuals`](@ref).
`block > 0` fixes the scratch-block width (default: a 64 MiB budget).

!!! warning
    This overwrites the Krylov basis: call it only once the factorization is finished.
"""
function extract_eigenpairs!(factorization::KrylovFactorization, ϕ::AbstractMatrix,
                             idxs::AbstractVector, mapvals; block::Int=0)
    basis = factorization.basis
    # An empty result gets its own 0-size matrix rather than a view that would pin the basis.
    (basis isa MatrixBasis && !isempty(idxs)) ||
        return calc_eigen_residuals(factorization, ϕ, idxs, mapvals)
    layout = factorization.layout
    vecs = _reconstruct_in_place!(layout, basis.basis, basis.nvecs, ϕ[:, idxs], block)
    vals, residuals = build_values_and_residuals(layout, vecs, mapvals, factorization.pu)
    return vals, vecs, residuals
end

# Eigenvector block J needs only basis block J (columns for row layout, rows for column
# layout), so each block is computed into scratch and written back over the basis slots it has
# just consumed. The leading `n ≤ k` slots end up holding the eigenvectors.
function _reconstruct_in_place!(::RowLayout, B::AbstractMatrix, k::Int, ϕs::AbstractMatrix,
                                block::Int)
    n, dim = size(ϕs, 2), size(B, 2)
    width = block > 0 ? min(block, dim) : _scratch_width(n, dim, eltype(B))
    scratch = similar(B, n, width)
    for j in 1:width:dim
        J = j:min(j + width - 1, dim)
        s = view(scratch, :, 1:length(J))
        mul!(s, ϕs', view(B, 1:k, J))              # n×|J| = (n×k)(k×|J|)
        view(B, 1:n, J) .= s
    end
    return view(B, 1:n, :)
end

function _reconstruct_in_place!(::ColumnLayout, B::AbstractMatrix, k::Int, ϕs::AbstractMatrix,
                                block::Int)
    n, dim = size(ϕs, 2), size(B, 1)
    height = block > 0 ? min(block, dim) : _scratch_width(n, dim, eltype(B))
    scratch = similar(B, height, n)
    for i in 1:height:dim
        I = i:min(i + height - 1, dim)
        s = view(scratch, 1:length(I), :)
        mul!(s, view(B, I, 1:k), ϕs)                # |I|×n = (|I|×k)(k×n)
        view(B, I, 1:n) .= s
    end
    return view(B, :, 1:n)
end

"""
    permute_eigenvectors!(layout, vecs, perm; block=0) -> vecs

Reorder the eigenvectors in place — eigenvector `i` becomes old eigenvector `perm[i]` — through
one scratch block instead of a permuted copy. `vecs` must be a matrix or a view of its leading
rows (row layout) / columns (column layout), as [`extract_eigenpairs!`](@ref) returns.
"""
function permute_eigenvectors!(layout::Layout, vecs::AbstractMatrix, perm::AbstractVector{<:Integer};
                               block::Int=0)
    n = length(perm)
    n == _neigvecs(layout, vecs) || throw(DimensionMismatch(
        "perm has $n entries for $(_neigvecs(layout, vecs)) eigenvectors"))
    (n <= 1 || issorted(perm)) && return vecs          # a sorted permutation is the identity
    M = parent(vecs)
    _is_leading(layout, vecs, M, n) ||
        throw(ArgumentError("vecs must be a matrix or a view of its leading eigenvector slots"))
    # Gather through the parent with absolute indices (no view-of-view on a device array).
    _permute_leading!(layout, M, n, is_gpu_array(M) ? gpu_vector(perm) : perm, block)
    return vecs
end

_is_leading(::RowLayout, vecs, M, n) =
    size(vecs, 2) == size(M, 2) && parentindices(vecs)[1] == 1:n
_is_leading(::ColumnLayout, vecs, M, n) =
    size(vecs, 1) == size(M, 1) && parentindices(vecs)[2] == 1:n

# Row layout is column-major `n × dim`, so each column block holds a contiguous slice of every
# eigenvector: gathering block by block streams memory instead of striding along whole rows.
function _permute_leading!(::RowLayout, M::AbstractMatrix, n::Int, idx::AbstractVector, block::Int)
    dim = size(M, 2)
    width = block > 0 ? min(block, dim) : _scratch_width(n, dim, eltype(M))
    scratch = similar(M, n, width)
    for j in 1:width:dim
        J = j:min(j + width - 1, dim)
        s = view(scratch, :, 1:length(J))
        s .= view(M, idx, J)
        view(M, 1:n, J) .= s
    end
    return nothing
end

function _permute_leading!(::ColumnLayout, M::AbstractMatrix, n::Int, idx::AbstractVector, block::Int)
    dim = size(M, 1)
    height = block > 0 ? min(block, dim) : _scratch_width(n, dim, eltype(M))
    scratch = similar(M, height, n)
    for i in 1:height:dim
        I = i:min(i + height - 1, dim)
        s = view(scratch, 1:length(I), :)
        s .= view(M, I, idx)
        view(M, I, 1:n) .= s
    end
    return nothing
end
