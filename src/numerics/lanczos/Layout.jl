# Block storage orientation for Lanczos. A block of `bs` Krylov vectors can be stored
# with the vectors as COLUMNS (`D × bs`, operator action `H·X`) or as ROWS (`bs × D`,
# operator action `X·H`). The row layout makes the per-block inner loop contiguous in
# memory, which is substantially faster for sparse `H` (see
# `src/UniPolfed/Optimization/NOTES.md`). Both are mathematically equivalent for a
# Hermitian operator; only the storage axis and the side of each matrix operation differ.
#
# The orientation is carried as a singleton trait so dispatch is fully static. Every
# orientation-sensitive operation in the factorization goes through one of the seven
# primitives below; the rest of Lanczos is layout-agnostic.

"""
    Layout

Abstract trait selecting how a block of Krylov vectors is stored: [`ColumnLayout`](@ref)
(vectors as columns, `D × bs`) or [`RowLayout`](@ref) (vectors as rows, `bs × D`).
"""
abstract type Layout end

"""    ColumnLayout() — block stored as `D × bs`; vectors are columns; operator action `H·X`."""
struct ColumnLayout <: Layout end

"""    RowLayout() — block stored as `bs × D`; vectors are rows; operator action `X·H`."""
struct RowLayout <: Layout end

# --- shape -------------------------------------------------------------------

"""    blocksize(layout, x) -> Int — number of vectors in the block `x`."""
@inline blocksize(::ColumnLayout, x::AbstractMatrix) = size(x, 2)
@inline blocksize(::RowLayout,    x::AbstractMatrix) = size(x, 1)

"""    opdim(layout, x) -> Int — the operator (Hilbert-space) dimension of the block `x`."""
@inline opdim(::ColumnLayout, x::AbstractMatrix) = size(x, 1)
@inline opdim(::RowLayout,    x::AbstractMatrix) = size(x, 2)

# A plain vector is a single Krylov vector, orientation-free.
@inline blocksize(::Layout, ::AbstractVector) = 1
@inline opdim(::Layout, x::AbstractVector) = length(x)

# --- orthonormalization ------------------------------------------------------

"""
    orthonormalize(layout, X) -> (Q, β)

Orthonormalize the block `X` (same shape out). Returns the orthonormal block `Q` and the
`bs × bs` triangular coupling factor `β` such that `X = Q·β` (column: `qr`, orthonormal
columns, `β` upper-triangular `R`) or `X = β·Q` (row: `lq`, orthonormal rows, `β`
lower-triangular `L`). `β` is the block analogue of the Lanczos norm coupling.
"""
function orthonormalize(::ColumnLayout, X::AbstractMatrix)
    F = qr(X)
    return Matrix(F.Q), Matrix(F.R)
end
function orthonormalize(::RowLayout, X::AbstractMatrix)
    F = lq(X)
    return Matrix(F.Q), Matrix(F.L)
end

# --- overlap (Gram block) ----------------------------------------------------

"""
    overlap(layout, V, R) -> Matrix

Block overlap (Gram matrix) between two blocks `V` and `R`: `[i,j] = ⟨vᵢ, rⱼ⟩` with the
vectors taken per `layout`. For orthonormal `V` this is also the projection coefficient
used by [`project!`](@ref); it doubles as the orthogonality probe (`≈ 0` ⇔ `R ⟂ V`).
"""
@inline overlap(::ColumnLayout, V::AbstractMatrix, R::AbstractMatrix) = V' * R
@inline overlap(::RowLayout,    V::AbstractMatrix, R::AbstractMatrix) = R * V'

# --- projection (orthogonalization step) -------------------------------------

"""
    project!(layout, R, V, α) -> R

Subtract the `V`-component from `R` in place: `R ← R − V·α` (column) or `R ← R − α·V`
(row), where `α = overlap(layout, V, R)` for orthonormal `V`. After
`project!(layout, R, V, overlap(layout, V, R))`, `overlap(layout, V, R) ≈ 0`.
"""
@inline project!(::ColumnLayout, R::AbstractMatrix, V::AbstractMatrix, α::AbstractMatrix) =
    mul!(R, V, α, -1, true)
@inline project!(::RowLayout, R::AbstractMatrix, V::AbstractMatrix, α::AbstractMatrix) =
    mul!(R, α, V, -1, true)

# --- reorthogonalization against an accumulated basis ------------------------

"""
    reorthogonalize!(layout, W, Q) -> W

Remove the component of `W` in the span of the orthonormal basis block `Q` (which holds
`k` already-converged Krylov vectors), in place: `W ← W − Q(Q'W)` (column) or
`W ← W − (WQ')Q` (row). After the call, `overlap(layout, Q, W) ≈ 0`.
"""
@inline function reorthogonalize!(::ColumnLayout, W::AbstractVecOrMat, Q::AbstractMatrix)
    W .-= Q * (Q' * W)
    return W
end
@inline function reorthogonalize!(::RowLayout, W::AbstractMatrix, Q::AbstractMatrix)
    W .-= (W * Q') * Q
    return W
end

# --- eigenvector reconstruction ----------------------------------------------

"""
    reconstruct(layout, basis, ϕ) -> Matrix

Reconstruct the eigenvectors in the original space from the Krylov `basis` (`k` vectors)
and the Ritz coefficient matrix `ϕ` (`k × m`). Column layout returns `basis·ϕ` (`D × m`,
eigenvectors as columns); row layout returns `ϕ'·basis` (`m × D`, eigenvectors as rows).
"""
@inline reconstruct(::ColumnLayout, basis::AbstractMatrix, ϕ::AbstractMatrix) = basis * ϕ
@inline reconstruct(::RowLayout,    basis::AbstractMatrix, ϕ::AbstractMatrix) = ϕ' * basis
