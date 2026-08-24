# universal — construction. Turns (variant, structure, block size, params) into the row-layout
# closure. Dispatch is on `Val` of the variant, so a second universal kernel is one more method
# here plus one more file under kernels/ — nothing else changes.

"""
    build_kernel(::Val{:universal}, ::Val{:row_read}, s::MappingStructure, ::Val{bs})

The `universal/row_read` closure `(Y, X) -> nothing` over the structure `s`, specialized to block
size `bs`. `Y` and `X` are `bs × dim`, and the closure computes `Y = (H · Xᵀ)ᵀ`.

Runs on any structure. Takes no tuning parameters.
"""
build_kernel(::Val{:universal}, ::Val{:row_read}, s::MappingStructure, ::Val{bs}) where {bs} =
    _build_row_read(s, Val(bs))

"""
    build_kernel(::Val{:universal}, ::Val{:row_bucketed}, s::MappingStructure, ::Val{bs})

The `universal/row_bucketed` closure. Same result as `row_read`, reached by grouping rows of equal
neighbour count so the gather trip count is a compile-time constant.

Requires ONE value group, and is worth choosing only at `bs = 1` — see the kernel file for the
measurements and [`can_apply`](@ref) for the restriction. Takes no tuning parameters.
"""
build_kernel(::Val{:universal}, ::Val{:row_bucketed}, s::MappingStructure, ::Val{bs}) where {bs} =
    _build_row_bucketed(s, Val(bs))

# ── range forms ───────────────────────────────────────────────────────────────────────────────

supports_range(::Val{:universal}, ::Val{:row_read}) = true
# row_bucketed deliberately does NOT: it opens one parallel region per length bucket plus a tail,
# so its work is not a single flat range. It is only auto-selected at bs = 1 anyway.

"""    build_range_kernel(::Val{:universal}, ::Val{:row_read}, s, ::Val{bs}) — serial range over rows."""
build_range_kernel(::Val{:universal}, ::Val{:row_read}, s::MappingStructure, ::Val{bs}) where {bs} =
    _build_row_read_range(s, Val(bs))

range_units(::Val{:universal}, ::Val{:row_read}, s::MappingStructure, ::Val) = s.dim
