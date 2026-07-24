# Problem context: the shape/type/device metadata every POLFED stage needs to size
# probes and buffers. This is derived once from the starting vector/block `x0` (which
# does double duty — it is both this template and the Lanczos seed) and then carried on
# the MappingPlan (`mp.sys`), so downstream stages take just the plan, not a tag-along
# `x0`. The element type is a *type parameter* (not a stored field), so it is available
# through the struct's type without boxing.

"""
    SysInfo{E,PU,X}

Immutable bundle of the POLFED problem's shape, element type, and processing unit,
derived from the starting vector/block `x0`. Held on the [`MappingPlan`](@ref) as
`mp.sys`.

Constructed by [`SysInfo`](@ref)`(x0, pu)` (called inside [`build_mapping_plan`](@ref)),
where the axes of `x0` fix `dim`/`block_size`.

# Fields
- `x0::X`: the starting vector/block, stored **by reference** (not copied) — it is also
  the Lanczos seed.
- `dim::Int`: Hilbert-space dimension — the **larger** axis of `x0`.
- `block_size::Int`: number of Krylov vectors — the **smaller** axis of `x0` (`1` for a
  vector).
- `pu::PU`: the [`CPU`](@ref)/[`GPU`](@ref) backend `x0` lives on.

The element type `E = eltype(x0)` is a type parameter, read via `eltype(sys)`, not a
stored field.

# Notes
`dim`/`block_size` are taken as `max`/`min` of the axes rather than a fixed
`(size(x0,1), size(x0,2))`, so a block is identified as the smaller axis regardless of
storage orientation (column- vs row-major); a full [`Layout`](@ref)-aware split lands
with the optimization slice.
"""
struct SysInfo{E<:Number,PU<:ProcessingUnit,X<:AbstractVecOrMat{E}}
    x0::X
    dim::Int
    block_size::Int
    pu::PU
end

"""
    _shape(x0) -> (dim, block_size)

Hilbert dimension and block size for a starting vector/block: `(length, 1)` for a
vector, `(max axis, min axis)` for a matrix — the smaller axis is always the block.
"""
_shape(x0::AbstractVector) = (length(x0), 1)
function _shape(x0::AbstractMatrix)
    d1, d2 = size(x0)
    return max(d1, d2), min(d1, d2)
end

"""
    SysInfo(x0, pu::ProcessingUnit) -> SysInfo

Build a [`SysInfo`](@ref) from the starting vector/block `x0` and its processing unit,
storing `x0` by reference and reading `dim`/`block_size` from its axes.
"""
function SysInfo(x0::AbstractVecOrMat{E}, pu::PU) where {E<:Number,PU<:ProcessingUnit}
    dim, block_size = _shape(x0)
    return SysInfo{E,PU,typeof(x0)}(x0, dim, block_size, pu)
end

"""    eltype(sys::SysInfo) -> Type — the element type `E` of `x0`, from the struct's type."""
Base.eltype(::Type{<:SysInfo{E}}) where {E} = E
Base.eltype(sys::SysInfo) = eltype(typeof(sys))
