# Row-layout operator builders for the matrix entrypoint `polfed(H, …)`. When POLFED owns a concrete
# `H`, it runs Lanczos row-native (the faster layout for the value-grouped kernels), so it needs the
# rescaled transform operator and the map-back as ROW operators (`Y ← X·H`). These helpers build
# them; they are private to `polfed(H, …)`.

"""
    _generic_row_rescaled(H, a, b) -> Function

A generic **row**-layout rescaled transform operator `Y ← X·H̃ = (X·H - b·X)/a` over a `bs × D`
block, via `mul!(Y, X, H)` then the affine rescale. The `optimize_mapping = false` fallback for the
matrix entrypoint's `f!_rescaled` (a matrix not structured enough for the fast kernels that should
still run row-native). Shape-generic — one closure serves every block size.
"""
function _generic_row_rescaled(H::AbstractMatrix, a::Real, b::Real)
    return @inline (Y::AbstractMatrix, X::AbstractMatrix) -> begin
        mul!(Y, X, H)
        @. Y = (Y - b * X) / a
        return nothing
    end
end

"""
    _row_mapback(H) -> Function

The row-layout map-back operator `Y ← X·H` via `mul!(Y, X, H)`, used as the Lanczos `mapvals`.
Deliberately **generic** (not a value-grouped kernel): map-back is applied only a handful of times
per convergence check, to a *single* strided `1 × D` eigenvector view — where the fast SIMD kernels
give no benefit and the `:shuf` kernel cannot even load (it needs contiguous memory). `mul!` handles
the strided view, any block size, complex, and GPU alike.
"""
_row_mapback(H::AbstractMatrix) =
    @inline (Y::AbstractMatrix, X::AbstractMatrix) -> (mul!(Y, X, H); nothing)

"""
    _build_matrix_rowops(H, choice, bs::Int, a, b) -> (; f!_rescaled, f!)

Resolve the row-layout operators for the matrix entrypoint from the `optimize_mapping` request
`choice`:

- `f!_rescaled` (the transform, POLFED's bottleneck): the fast value-grouped `bs × D` block kernel
  ([`build_optimized_mapping`](@ref)) when `choice` is truthy, else a generic `mul!` row operator
  ([`_generic_row_rescaled`](@ref));
- `f!` (the one-time map-back `mapvals`): always the generic `mul!` row operator
  ([`_row_mapback`](@ref)), regardless of `choice`.

`bs` is the Krylov block size (runtime); it is lifted to `Val(bs)` here (one-time, at setup) so the
transform kernel closure stays type-stable in the hot loop.
"""
function _build_matrix_rowops(H::AbstractMatrix, choice, bs::Int, a::Real, b::Real)
    f! = _row_mapback(H)
    f!_rescaled = choice === false ? _generic_row_rescaled(H, a, b) :
                  build_optimized_mapping(H, choice, Val(bs), a, b).f!_rescaled
    return (; f!_rescaled, f!)
end
