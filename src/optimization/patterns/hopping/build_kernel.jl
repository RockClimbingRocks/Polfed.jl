# hopping — construction.
#
# The pipeline's currency is `MappingStructure`, so both methods take one and recover the lattice
# with `_derive_hop`. `can_apply` has already established that this succeeds; the `isnothing`
# branch is a guard against a caller reaching past it.

"""
    build_kernel(::Val{:hopping}, ::Val{:packed}, s::MappingStructure, ::Val{bs})

The `hopping/packed` closure `(Y, X) -> nothing`: site-packed SIMD with the axis-1 neighbours
taken from registers. The default for this pattern — see the kernel file for the measurements.

Requires a real element type and dense `Matrix` arguments; `bs` must divide the pack width.
[`can_apply`](@ref) covers the first, `_build_hop_packed` throws on the rest.
"""
function build_kernel(::Val{:hopping}, ::Val{:packed}, s::MappingStructure, ::Val{bs};
                      W::Int = HOP_PACKED_PARAMS.W,
                      tile::Int = HOP_PACKED_PARAMS.tile) where {bs}
    hs = _derive_hop(s)
    isnothing(hs) && throw(ArgumentError(
        "hopping/packed: this matrix is not a periodic nearest-neighbour hopping operator."))
    return _build_hop_packed(hs, Val(bs); W = W, tile = tile)
end

"""
    build_kernel(::Val{:hopping}, ::Val{:rows}, s::MappingStructure, ::Val{bs})

The `hopping/rows` closure `(Y, X) -> nothing`: `SVector{bs}` registers slid along axis 1. The
fallback — any `bs`, any element type, any `AbstractMatrix`.
"""
function build_kernel(::Val{:hopping}, ::Val{:rows}, s::MappingStructure, ::Val{bs};
                      tile::Int = HOP_ROWS_PARAMS.tile) where {bs}
    hs = _derive_hop(s)
    isnothing(hs) && throw(ArgumentError(
        "hopping/rows: this matrix is not a periodic nearest-neighbour hopping operator."))
    return _build_hop_rows(hs, Val(bs); tile = tile)
end

"""
    build_hop_kernel(hs::HopStructure, ::Val{bs}; variant = :auto, W = …, tile = …) -> Function

Build a hopping matvec **without a matrix** — the matrix-free entry point.

This is where the memory win lands: a 128³ Anderson operator is ~1.1 GB of `SparseMatrixCSC`,
assembled only to be classified and thrown away, and a user who already knows their lattice
should never pay it. Pair it with `MappingConfig(f!_rescaled = …)`, having rescaled the structure
first:

```julia
hs = HopStructure(randn(prod(shape)), shape, -1.0)
f! = build_hop_kernel(rescale(hs, a, b), Val(bs))
```

`variant` defaults to `:auto`, which picks `packed` when its preconditions hold and `rows`
otherwise. `W` and `tile` are the chosen kernel's tuning parameters — see
[`HOP_PACKED_PARAMS`](@ref) and [`HOP_ROWS_PARAMS`](@ref); `W` is ignored by `rows`, which has no
pack width of its own.

# Throws
- `ArgumentError` for an unknown `variant`, or from the chosen kernel's own preconditions.
"""
function build_hop_kernel(hs::HopStructure{T,D}, ::Val{bs}; variant::Symbol = :auto,
                          W::Int = HOP_PACKED_PARAMS.W,
                          tile::Int = HOP_PACKED_PARAMS.tile) where {T,D,bs}
    if variant === :auto
        variant = _hop_packed_ok(T, bs) ? :packed : :rows
    end
    variant === :packed && return _build_hop_packed(hs, Val(bs); W = W, tile = tile)
    variant === :rows && return _build_hop_rows(hs, Val(bs); tile = tile)
    throw(ArgumentError("build_hop_kernel: unknown variant :$variant (use :packed or :rows)."))
end

# ── range forms ───────────────────────────────────────────────────────────────────────────────
# The hopping unit is a slice of one lattice line, not a basis state: these kernels partition the
# lattice, so `range_units` is `nlines × xsplit` rather than `dim`. Distinct unit ranges still
# write disjoint parts of `Y`, which is what a decomposition actually needs.

supports_range(::Val{:hopping}, ::Val{:packed}) = true
supports_range(::Val{:hopping}, ::Val{:rows})   = true

"""    build_range_kernel(::Val{:hopping}, ::Val{:packed}, s, ::Val{bs}; W) — serial range over work units."""
function build_range_kernel(::Val{:hopping}, ::Val{:packed}, s::MappingStructure, ::Val{bs};
                            W::Int = HOP_PACKED_PARAMS.W, tile::Int = HOP_PACKED_PARAMS.tile) where {bs}
    hs = _derive_hop(s)
    isnothing(hs) && throw(ArgumentError("hopping/packed: this matrix is not a hopping operator."))
    return _build_hop_packed_range(hs, Val(bs); W = W)
end

"""    build_range_kernel(::Val{:hopping}, ::Val{:rows}, s, ::Val{bs}) — serial range over work units."""
function build_range_kernel(::Val{:hopping}, ::Val{:rows}, s::MappingStructure, ::Val{bs};
                            tile::Int = HOP_ROWS_PARAMS.tile) where {bs}
    hs = _derive_hop(s)
    isnothing(hs) && throw(ArgumentError("hopping/rows: this matrix is not a hopping operator."))
    return _build_hop_rows_range(hs, Val(bs))
end

function range_units(::Val{:hopping}, ::Val{:packed}, s::MappingStructure, ::Val{bs};
                     W::Int = HOP_PACKED_PARAMS.W, tile::Int = HOP_PACKED_PARAMS.tile) where {bs}
    hs = _derive_hop(s)
    isnothing(hs) && throw(ArgumentError("hopping/packed: this matrix is not a hopping operator."))
    return _hop_packed_units(hs, Val(bs); W = W)
end

function range_units(::Val{:hopping}, ::Val{:rows}, s::MappingStructure, ::Val{bs};
                     tile::Int = HOP_ROWS_PARAMS.tile) where {bs}
    hs = _derive_hop(s)
    isnothing(hs) && throw(ArgumentError("hopping/rows: this matrix is not a hopping operator."))
    return _hop_rows_units(hs, Val(bs))
end
