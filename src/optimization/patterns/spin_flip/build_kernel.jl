# spin_flip — construction.
#
# The pipeline's currency is `MappingStructure`, so both methods take one and recover the term
# list with `_derive_flip`. `can_apply` has already established that this succeeds; the
# `isnothing` branch is a guard against a caller reaching past it.

"""
    build_kernel(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure, ::Val{bs})

The `spin_flip/packed` closure `(Y, X) -> nothing`: column-packed SIMD, masks lifted into the
type domain. The default for this pattern — see the kernel file for the measurements.

Requires a real element type, a power-of-two `bs` no wider than the pack, and dense `Matrix`
arguments. [`can_apply`](@ref) covers the first; `_build_packed` throws on the second.
"""
function build_kernel(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure, ::Val{bs};
                     W::Int = PACKED_PARAMS.W, G::Int = PACKED_PARAMS.G) where {bs}
    fs = _derive_flip(s)
    isnothing(fs) && throw(ArgumentError(
        "spin_flip/packed: this matrix is not a spin-flip operator."))
    return _build_packed(fs, Val(bs); W = W, G = G)
end

"""
    build_kernel(::Val{:spin_flip}, ::Val{:rows}, s::MappingStructure, ::Val{bs})

The `spin_flip/rows` closure `(Y, X) -> nothing`: `SVector{bs}` accumulators over a runtime
mask list. The fallback — any `bs`, any element type, any `AbstractMatrix`.
"""
function build_kernel(::Val{:spin_flip}, ::Val{:rows}, s::MappingStructure, ::Val{bs};
                     G::Int = ROWS_PARAMS.G) where {bs}
    fs = _derive_flip(s)
    isnothing(fs) && throw(ArgumentError(
        "spin_flip/rows: this matrix is not a spin-flip operator."))
    return _build_rows(fs, Val(bs); G = G)
end

"""
    build_flip_kernel(fs::FlipStructure, ::Val{bs}; variant = :auto, W = …, G = …) -> Function

Build a spin-flip matvec **without a matrix** — the matrix-free entry point.

This is where the memory win lands: assembling `H` just to classify it costs ~350 MB at L=20
and ~5.6 GB at L=24, and a user who already knows their model's masks should never pay it.
Pair it with `MappingConfig(f!_rescaled = …)`, having rescaled the structure first:

```julia
fs = FlipStructure(diags, masks, values)
f! = build_flip_kernel(rescale(fs, a, b), Val(bs))
```

`variant` defaults to `:auto`, which picks `packed` when its preconditions hold and `rows`
otherwise. `W`/`G` are the chosen kernel's tuning parameters — see [`PACKED_PARAMS`](@ref) and
[`ROWS_PARAMS`](@ref).
"""
function build_flip_kernel(fs::FlipStructure{T}, ::Val{bs}; variant::Symbol = :auto,
                           W::Int = PACKED_PARAMS.W, G::Int = -1) where {T,bs}
    if variant === :auto
        variant = _packed_ok(T, bs, fs.dim) ? :packed : :rows
    end
    variant === :packed &&
        return _build_packed(fs, Val(bs); W = W, G = G < 0 ? PACKED_PARAMS.G : G)
    variant === :rows &&
        return _build_rows(fs, Val(bs); G = G < 0 ? ROWS_PARAMS.G : G)
    throw(ArgumentError("build_flip_kernel: unknown variant :$variant (use :packed or :rows)."))
end

# ── range forms ───────────────────────────────────────────────────────────────────────────────

supports_range(::Val{:spin_flip}, ::Val{:packed}) = true
supports_range(::Val{:spin_flip}, ::Val{:rows})   = true

"""
    build_range_kernel(::Val{:spin_flip}, ::Val{:packed}, s, ::Val{bs}; W, G)

The serial range closure over basis states `lo:hi`. `lo - 1` must be a multiple of
[`range_alignment`](@ref) — the kernel steps in whole packs.
"""
function build_range_kernel(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure, ::Val{bs};
                            W::Int = PACKED_PARAMS.W, G::Int = PACKED_PARAMS.G) where {bs}
    fs = _derive_flip(s)
    isnothing(fs) && throw(ArgumentError("spin_flip/packed: this matrix is not a spin-flip operator."))
    return _build_packed_range(fs, Val(bs); W = W, G = G)
end

"""    build_range_kernel(::Val{:spin_flip}, ::Val{:rows}, s, ::Val{bs}; G) — serial range over columns."""
function build_range_kernel(::Val{:spin_flip}, ::Val{:rows}, s::MappingStructure, ::Val{bs};
                            G::Int = ROWS_PARAMS.G) where {bs}
    fs = _derive_flip(s)
    isnothing(fs) && throw(ArgumentError("spin_flip/rows: this matrix is not a spin-flip operator."))
    return _build_rows_range(fs, Val(bs); G = G)
end

range_units(::Val{:spin_flip}, ::Val, s::MappingStructure, ::Val; kwargs...) = s.dim

"""
    range_alignment(::Val{:spin_flip}, ::Val{:packed}, s, ::Val{bs}; W, G) -> Int

`P·G = (W÷bs)·G`, with `W` derived from the hardware when `0` and `G` shrunk exactly as
[`_build_packed`](@ref) shrinks it, so the number matches what the range closure will enforce.
"""
function range_alignment(::Val{:spin_flip}, ::Val{:packed}, s::MappingStructure, ::Val{bs};
                         W::Int = PACKED_PARAMS.W, G::Int = PACKED_PARAMS.G) where {bs}
    fs = _derive_flip(s)
    isnothing(fs) && throw(ArgumentError("spin_flip/packed: this matrix is not a spin-flip operator."))
    W = W == 0 ? _packed_lane_width(eltype(fs.diags)) : W
    P = W ÷ bs
    while G > 1 && fs.dim < P * G
        G >>= 1
    end
    return P * G
end
