# spin_flip / rows — the general fallback: SVector{bs} accumulators, RUNTIME mask list.
#
#   Y[:, i] = ε[i]·X[:, i] + Σ_k v_k · X[:, xor(i-1, mask_k) + 1]
#
# `G` output columns are accumulated at once, interleaved INSIDE the shared mask loop so the G
# dependency chains overlap. That is the only form of the trick that works: running each
# column's loop to completion in turn leaves the compiler nothing to interleave, and building
# the G accumulators with `ntuple(g -> ..., Val(G))` heap-allocates (the defect that made
# `universal/row_read` 8-12x slower than this shape before it was removed).
#
# This kernel exists for everything `packed` cannot take: `bs` not a power of two, `bs > W`, a
# non-`Matrix` argument, or a complex element type (SIMD.jl `Vec` carries no `Complex`).
#
# The mask list stays a RUNTIME `Vector`. A compile-time-unrolled variant was measured and is
# NOT used here: it is 24% faster at bs=1 but collapses past it, because the M x G fully
# unrolled term sum spills registers (QSun L=20 t=1, per-column ms):
#
#             bs1     bs2     bs4     bs8
#   runtime   8.391   3.945   2.465   1.951
#   unrolled  6.379   5.283   6.901   7.971
#
# `packed` already owns bs=1, so the fallback never needs the unroll.

"""
    ROWS_PARAMS

This kernel's tunable parameters and their measured defaults.

- `G`: output columns accumulated in flight. `G = 8` was best across the spin_flip sweep, but
  the optimum is a register-pressure trade-off — it moves with `bs`, the element type and the
  machine, so it is exposed rather than fixed. Sweep it with `benchmark_mapping`, or pin it:
  `MappingKernel(:spin_flip, :rows; G = 16)`.
"""
const ROWS_PARAMS = (G = 8,)

"""    _flip_loadv(X, j, ::Val{bs}, ::Type{T}) -> SVector{bs,T} — column `j` of a row block."""
@inline _flip_loadv(X::AbstractMatrix, j::Int, ::Val{bs}, ::Type{T}) where {bs,T} =
    SVector{bs,T}(ntuple(b -> @inbounds(X[b, j]), Val(bs)))

"""    _flip_storev!(Y, i, acc) — write the `bs` amplitudes of column `i` back."""
@inline function _flip_storev!(Y::AbstractMatrix, i::Int, acc::SVector{bs}) where {bs}
    @inbounds for b in 1:bs
        Y[b, i] = acc[b]
    end
    return nothing
end

"""
    _rows_block!(Y, X, diags, masks, values, i0, ::Val{bs}, ::Val{G})

Accumulate `G` consecutive output columns starting at `i0`. Generated so the `G` accumulators
are plain locals — no tuple is materialised, which is what keeps this allocation-free.
"""
@generated function _rows_block!(Y, X, diags::Vector{T}, masks::Vector{Int}, values::Vector{T},
                                 i0::Int, ::Val{bs}, ::Val{G}) where {T,bs,G}
    init, upd, store = Expr[], Expr[], Expr[]
    for g in 0:G-1
        a, p = Symbol(:a_, g), Symbol(:p_, g)
        push!(init, :($a = @inbounds(diags[i0 + $g]) * _flip_loadv(X, i0 + $g, Val(bs), T)))
        push!(init, :($p = i0 + $g - 1))                      # 0-based state index
        push!(upd,  :($a = muladd(v, _flip_loadv(X, xor($p, m) + 1, Val(bs), T), $a)))
        push!(store, :(_flip_storev!(Y, i0 + $g, $a)))
    end
    quote
        $(init...)
        @inbounds for k in eachindex(masks)
            m = masks[k]; v = values[k]
            $(upd...)
        end
        $(store...)
        return nothing
    end
end

"""
    _rows_range!(Y, X, diags, masks, values, lo, hi, ::Val{bs}, ::Val{G})

Run columns `lo:hi`, `G` at a time, then the tail one at a time. The tail is not optional: a
missed one leaves stale data in `Y`, which surfaces much later as a wrong eigenvector.
"""
function _rows_range!(Y, X, diags::Vector{T}, masks::Vector{Int}, values::Vector{T},
                      lo::Int, hi::Int, ::Val{bs}, ::Val{G}) where {T,bs,G}
    i = lo
    while i + G - 1 <= hi
        _rows_block!(Y, X, diags, masks, values, i, Val(bs), Val(G))
        i += G
    end
    while i <= hi
        _rows_block!(Y, X, diags, masks, values, i, Val(bs), Val(1))
        i += 1
    end
    return nothing
end

"""
    _rows_apply!(Y, X, fs::FlipStructure, ::Val{bs})

One parallel region per apply, chunked over columns — never indexed by `threadid()` (tasks
migrate on Julia >= 1.12). The single-thread path skips the parallel region entirely so the
kernel stays allocation-free when there is nothing to parallelise.
"""
function _rows_apply!(Y::AbstractMatrix{T}, X::AbstractMatrix{T},
                      fs::FlipStructure{T}, ::Val{bs}, Gv::Val) where {T,bs}
    D = fs.dim
    diags = fs.diags; masks = fs.masks; values = fs.values
    if Threads.nthreads() == 1
        _rows_range!(Y, X, diags, masks, values, 1, D, Val(bs), Gv)
        return nothing
    end
    nt = max(1, min(Threads.nthreads(), cld(D, 64)))
    chunk = cld(D, nt)
    Threads.@threads :static for t in 1:nt
        _rows_range!(Y, X, diags, masks, values,
                     (t - 1) * chunk + 1, min(t * chunk, D), Val(bs), Gv)
    end
    return nothing
end

"""
    _build_rows(fs::FlipStructure, ::Val{bs}; G) -> Function

Close over the flip structure and return the row-layout matvec `(Y, X) -> nothing`. Accepts
any `bs`, any element type, and any `AbstractMatrix` argument.

`G` is lifted to a `Val` HERE, not inside the closure: building it from a runtime `Int` per
apply would be a dynamic dispatch on every matvec.
"""
function _build_rows(fs::FlipStructure{T}, ::Val{bs}; G::Int = ROWS_PARAMS.G) where {T,bs}
    G >= 1 || throw(ArgumentError("spin_flip/rows: G must be >= 1, got $G."))
    Gv = Val(G)
    return (Y, X) -> _rows_apply!(Y, X, fs, Val(bs), Gv)
end

"""
    _build_rows_range(fs::FlipStructure, ::Val{bs}; G) -> Function

The SERIAL range form `(Y, X, lo, hi) -> nothing` over columns `lo:hi` (1-based, inclusive),
for callers that own the threads. Columns are independent, so any range is valid —
[`range_alignment`](@ref) is `1` for this kernel.
"""
function _build_rows_range(fs::FlipStructure{T}, ::Val{bs}; G::Int = ROWS_PARAMS.G) where {T,bs}
    G >= 1 || throw(ArgumentError("spin_flip/rows: G must be >= 1, got $G."))
    Gv = Val(G)
    diags = fs.diags; masks = fs.masks; values = fs.values
    return (Y, X, lo::Int, hi::Int) ->
        _rows_range!(Y, X, diags, masks, values, lo, hi, Val(bs), Gv)
end
