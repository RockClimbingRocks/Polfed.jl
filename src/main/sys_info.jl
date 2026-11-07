# Problem context: the shape/type/device metadata every POLFED stage needs to size
# probes and buffers. This is derived once from the starting vector/block `x0` (which
# does double duty — it is both this template and the Lanczos seed) and then carried on
# the MappingPlan (`mp.sys`), so downstream stages take just the plan, not a tag-along
# `x0`. The element type is a *type parameter* (not a stored field), so it is available
# through the struct's type without boxing.

"""
    SysInfo{E,PU,X,L}

Immutable bundle of the POLFED problem's shape, element type, processing unit, and block
[`Layout`](@ref), derived from the starting vector/block `x0`. Held on the
[`MappingPlan`](@ref) as `mp.sys`.

Constructed by [`SysInfo`](@ref)`(x0, pu, layout)` (called inside
[`build_mapping_plan`](@ref)), where `layout` and the axes of `x0` fix `dim`/`block_size`.

# Fields
- `x0::X`: the starting vector/block, stored **by reference** (not copied) — it is also
  the Lanczos seed.
- `dim::Int`: Hilbert-space dimension — [`opdim`](@ref)`(layout, x0)`.
- `block_size::Int`: number of Krylov vectors — [`blocksize`](@ref)`(layout, x0)` (`1` for
  a vector).
- `pu::PU`: the [`CPU`](@ref)/[`GPU`](@ref) backend `x0` lives on.
- `layout::L`: how the block is stored — [`ColumnLayout`](@ref) (`D × bs`) or
  [`RowLayout`](@ref) (`bs × D`). This is the ONE place downstream stages read the
  orientation from; anything that slices a single Krylov vector out of `x0` must consult
  it rather than assuming an axis.

The element type `E = eltype(x0)` is a type parameter, read via `eltype(sys)`, not a
stored field.

# Notes
The shape comes from the layout trait, not from `max`/`min` of the axes. The old heuristic
("the block is the smaller axis") silently transposed the problem whenever `block_size`
exceeded `dim` — reachable with the tiny systems used in tests — and could not be correct
at all when the two are equal.
"""
struct SysInfo{E<:Number,PU<:ProcessingUnit,X<:AbstractVecOrMat{E},L<:Layout}
    x0::X
    dim::Int
    block_size::Int
    pu::PU
    layout::L
end

"""
    _shape(x0, layout::Layout) -> (dim, block_size)

Hilbert dimension and block size for a starting vector/block, read through the layout
trait: `(length, 1)` for a vector, `(opdim, blocksize)` for a matrix. A plain vector is a
single Krylov vector and so is orientation-free.
"""
_shape(x0::AbstractVecOrMat, layout::Layout) = (opdim(layout, x0), blocksize(layout, x0))

"""
    SysInfo(x0, pu::ProcessingUnit, layout::Layout = ColumnLayout()) -> SysInfo

Build a [`SysInfo`](@ref) from the starting vector/block `x0`, its processing unit, and the
block layout, storing `x0` by reference and reading `dim`/`block_size` through `layout`.

`layout` defaults to [`ColumnLayout`](@ref), which is what the operator entrypoints use; the
matrix entrypoints pass [`RowLayout`](@ref).
"""
function SysInfo(x0::AbstractVecOrMat{E}, pu::PU,
                 layout::L = ColumnLayout()) where {E<:Number,PU<:ProcessingUnit,L<:Layout}
    dim, block_size = _shape(x0, layout)
    return SysInfo{E,PU,typeof(x0),L}(x0, dim, block_size, pu, layout)
end

"""    eltype(sys::SysInfo) -> Type — the element type `E` of `x0`, from the struct's type."""
Base.eltype(::Type{<:SysInfo{E}}) where {E} = E
Base.eltype(sys::SysInfo) = eltype(typeof(sys))

"""
    _vector_slice(layout::Layout, A::AbstractVecOrMat, i::Int) -> AbstractVecOrMat

The `i`-th Krylov vector of the block `A`, **shaped the way that layout's operator takes it**:

- [`ColumnLayout`](@ref) → `A[:, i]`, a length-`D` vector; the column operators are `mul!(y, H, x)`.
- [`RowLayout`](@ref) → `A[i:i, :]`, a `1 × D` **matrix**; the row operators are `mul!(Y, X, H̃)`
  over a `bs × D` block, and a `bs = 1` block is still two-dimensional.
- a plain vector is a single Krylov vector already, returned as-is.

The `i:i` is not cosmetic. `view(A, i, :)` drops the axis and yields a 1-D view, which has no
`mul!` method against a `D × D` operator — that mismatch is why `VectorWise` on a row-layout
block failed with a `TaskFailedException` rather than an error anyone could read.

Every stage that pulls one vector out of a block goes through this. A caller that needs a bare
length-`D` vector regardless of layout (a Lanczos probe seed, say) wraps the result in `vec`.

!!! warning
    Under [`RowLayout`](@ref) the result is a **strided** view (stride `bs`, since Julia is
    column-major). Generic `mul!` operators accept it; the optimized packed kernels take raw
    pointers and require a dense `Matrix`, so they cannot. A caller needing a kernel-compatible
    per-vector buffer must materialize one.
"""
@inline _vector_slice(::Layout, x::AbstractVector, ::Int) = x
@inline _vector_slice(::ColumnLayout, A::AbstractMatrix, i::Int) = view(A, :, i)
@inline _vector_slice(::RowLayout, A::AbstractMatrix, i::Int) = view(A, i:i, :)
