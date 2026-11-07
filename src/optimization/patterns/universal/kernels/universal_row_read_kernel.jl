# universal / row_read — the value-grouped row kernel. Reads each group's own per-row extent,
# so it runs on ANY structure `classify` can produce.
#
#   y[i] = diags[i]·x[i] + Σ_groups value · Σ_{j ∈ row i of that group} x[j]
#
# It reads each group's own per-row extent, so several off-diagonal values and irregular rows
# (symmetry sectors) are handled by construction rather than by a special case. With one group and
# regular rows it degenerates to the classic single-value row kernel.
#
# Layout: `Y`, `X` are `bs × dim` (row layout) and the kernel computes `Y = (H · Xᵀ)ᵀ`. The block
# index is the FAST axis, so the `bs` amplitudes of a basis state are contiguous — one neighbour
# lookup serves all `bs` Krylov vectors.
#
# There is deliberately NO block/ILP knob here. An earlier version walked `G` rows at a time via
# `ntuple(g -> _row_accumulate(...), Val(G))`; that closure does not stack-allocate, so it heap-
# allocated ~841 KB per apply and ran 8-12x slower than this plain loop — slower even than the
# generic `mul!` it exists to beat. LLVM already unrolls this loop. Measured: removing it took an
# Sz-sector XXZ (dim 12870, bs=1) from 1.10 ms to 0.087 ms, and `mul!` is 0.302 ms.
#
# Rows here have a RUNTIME trip count (`starts[i+1] - starts[i]`), which is what stops the
# compiler from unrolling the inner gather. `universal/row_bucketed` buys that back by grouping
# rows of equal length; it is a separate kernel because it only wins at bs = 1.

"""
    _loadv(X, j, ::Val{bs}) -> SVector{bs}

The `bs` amplitudes of basis state `j` as a static vector. Contiguous in row layout.
"""
@inline _loadv(X::AbstractMatrix{T}, j::Int, ::Val{bs}) where {T,bs} =
    SVector(ntuple(r -> @inbounds(X[r, j]), Val(bs)))

"""    _storev!(Y, j, v) — write the `bs` amplitudes of basis state `j` back."""
@inline function _storev!(Y::AbstractMatrix{T}, j::Int, v::SVector{bs,T}) where {T,bs}
    @inbounds for r in 1:bs
        Y[r, j] = v[r]
    end
    return nothing
end

"""
    _row_accumulate(X, diags, groups, i, ::Val{bs}) -> SVector{bs}

One row of the product: the diagonal term plus, for every value group, that group's neighbour sum
scaled by the group's value. The inner range comes from the group's own `starts`, which is what
admits irregular rows.
"""
@inline function _row_accumulate(X::AbstractMatrix{T}, diags::Vector{T},
                                 groups::Vector{ValueGroup{T,TI}}, i::Int,
                                 ::Val{bs}) where {T,TI,bs}
    acc = @inbounds(diags[i]) * _loadv(X, i, Val(bs))
    @inbounds for g in groups
        s = zero(SVector{bs,T})
        for c in g.starts[i]:(g.starts[i + 1] - 1)
            s += _loadv(X, Int(g.flat[c]), Val(bs))
        end
        acc += g.value * s
    end
    return acc
end

"""
    _row_read_range!(Y, X, s, lo, hi, ::Val{bs})

Rows `lo:hi` (1-based, inclusive) of the product — the SERIAL core. Opens no parallel region,
so a caller that owns the threads can call it from inside its own; `Threads.@threads :static`
cannot nest, which is why the threading lives one level up in [`_row_read!`](@ref).
"""
function _row_read_range!(Y::AbstractMatrix{T}, X::AbstractMatrix{T},
                          s::MappingStructure{T,TI}, lo::Int, hi::Int,
                          ::Val{bs}) where {T,TI,bs}
    diags = s.diags
    groups = s.groups
    for i in lo:hi
        _storev!(Y, i, _row_accumulate(X, diags, groups, i, Val(bs)))
    end
    return nothing
end

"""
    _row_read!(Y, X, s, ::Val{bs})

Apply the value-grouped operator described by `s` to the row-layout block `X`, writing `Y`.

Rows are split into contiguous chunks, one task per chunk (no per-thread buffers, so nothing is
indexed by `threadid()`), and each chunk calls [`_row_read_range!`](@ref). This is the
threading policy only — all the work is in the serial core.
"""
function _row_read!(Y::AbstractMatrix{T}, X::AbstractMatrix{T},
                    s::MappingStructure{T,TI}, ::Val{bs}) where {T,TI,bs}
    D = s.dim
    nt = max(1, min(Threads.nthreads(), cld(D, 64)))
    chunk = cld(D, nt)

    if nt == 1
        _row_read_range!(Y, X, s, 1, D, Val(bs))
    else
        Threads.@threads :static for t in 1:nt
            _row_read_range!(Y, X, s, (t - 1) * chunk + 1, min(t * chunk, D), Val(bs))
        end
    end
    return nothing
end

"""
    _build_row_read(s::MappingStructure, ::Val{bs}) -> Function

Close over the structure and return the row-layout matvec `(Y, X) -> nothing`, specialized on the
block size so the hot loop is type-stable. This kernel takes no tuning parameters.
"""
_build_row_read(s::MappingStructure{T,TI}, ::Val{bs}) where {T,TI,bs} =
    (Y, X) -> _row_read!(Y, X, s, Val(bs))

"""
    _build_row_read_range(s::MappingStructure, ::Val{bs}) -> Function

The SERIAL range form `(Y, X, lo, hi) -> nothing` over rows `lo:hi`, for callers that own the
threads themselves. Rows are independent here, so any `lo`/`hi` is valid — see
[`range_alignment`](@ref), which is `1` for this kernel.
"""
_build_row_read_range(s::MappingStructure{T,TI}, ::Val{bs}) where {T,TI,bs} =
    (Y, X, lo::Int, hi::Int) -> _row_read_range!(Y, X, s, lo, hi, Val(bs))
