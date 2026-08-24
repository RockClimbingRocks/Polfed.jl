# universal / row_bucketed — rows grouped by their EXACT neighbour count.
#
# `row_read`'s inner gather has a runtime trip count (`starts[i+1] - starts[i]`), so the compiler
# cannot unroll it and every row pays a loop-carried branch. Bucketing rows of equal length turns
# that count into a compile-time constant: each bucket gets a fully unrolled body, and two rows are
# interleaved inside it so their gather chains overlap.
#
# Measured against `row_read` on Sz-sector XXZ (znver4, wall == process-CPU to 3 digits):
#
#            dim = 12,870    dim = 48,620
#   bs = 1      1.91x           1.88x        ← the reason this kernel exists
#   bs = 4      0.84x           0.82x        ← LOSES: see below
#
# Why it is a SEPARATE kernel rather than a change to `row_read`:
#
#   * It only wins at bs = 1. At bs >= 4 the `SVector{bs}` load already saturates the load ports
#     (~1.4 cycles/gather, near the hardware floor), so the unroll buys nothing while the row
#     permutation's scattered `Y` writes and `diags` reads cost 18%.
#   * It needs ONE value group. Bucketing a multi-value structure would key on the per-group
#     length TUPLE, which is combinatorial. `can_apply` refuses those.
#   * It costs a repacked copy of the neighbour table plus a row permutation, and one compiled
#     body per specialized length.
#
# The unroll is NOT a tuning knob: cheap alternatives that avoid codegen were measured and lost.
# Splitting the accumulator 2-way or 4-way inside `row_read` (no permutation, no specialization)
# came out SLOWER than the plain loop, so the gain here is the compile-time trip count, not the
# dependency chain. Interleaving 2 rows beat 1, 4 and 8.

"""
    _PAIR

Rows interleaved inside one unrolled bucket body. Measured on Sz-sector XXZ at bs = 1: 2 beats
4 (+4%) and 8 (+18%). Not exposed — it is a property of this loop, not a user choice.
"""
const _PAIR = 2

"""
    _MAX_SPECIALIZED_LENGTHS

How many distinct row lengths get their own compiled body. Each one costs a `@generated`
specialization per block size, so an unbounded count would make build time scale with the
Hamiltonian's row-length spread — and patched POLFED builds a kernel per patch. The lengths
covering the most rows are specialized; everything else falls to [`_TailBucket`](@ref), which
keeps the generic runtime-length loop.
"""
const _MAX_SPECIALIZED_LENGTHS = 12

"""
    _LengthBucket{T,L}

The rows having exactly `L` off-diagonal neighbours, with their neighbour columns repacked
contiguously. `L` is a type parameter, which is the whole point: it makes the gather trip count a
compile-time constant.

# Fields
- `rows::Vector{Int32}`: output row indices in this bucket.
- `nb::Matrix{Int32}`: `L × length(rows)`, column `k` holding row `rows[k]`'s neighbours.
- `value::T`: the group's shared off-diagonal value.
"""
struct _LengthBucket{T<:Number,L}
    rows::Vector{Int32}
    nb::Matrix{Int32}
    value::T
end

"""
    _TailBucket{T}

Rows whose length was not specialized (see [`_MAX_SPECIALIZED_LENGTHS`](@ref)), in CSR form with a
runtime trip count — i.e. exactly what `row_read` does, for the minority of rows that would
otherwise blow up compile time.

# Fields
- `rows::Vector{Int32}`: output row indices.
- `starts::Vector{Int32}`: length `length(rows) + 1`, indexing into `flat`.
- `flat::Vector{Int32}`: neighbour columns, concatenated in `rows` order.
- `value::T`: the group's shared off-diagonal value.
"""
struct _TailBucket{T<:Number}
    rows::Vector{Int32}
    starts::Vector{Int32}
    flat::Vector{Int32}
    value::T
end

"""
    _build_buckets(s::MappingStructure) -> (specialized::Tuple, tail::_TailBucket)

Partition `s`'s single value group's rows by neighbour count.

The lengths covering the most rows are specialized (up to `_MAX_SPECIALIZED_LENGTHS`), so the
compiled bodies are spent where they do the most work; the remainder go to the tail bucket. Every
row lands in exactly one bucket, so the kernel writes each output row exactly once.
"""
function _build_buckets(s::MappingStructure{T,TI}) where {T,TI}
    g = only(s.groups)
    D = s.dim
    lens = Vector{Int}(undef, D)
    @inbounds for i in 1:D
        lens[i] = g.starts[i + 1] - g.starts[i]
    end

    counts = Dict{Int,Int}()
    for n in lens
        counts[n] = get(counts, n, 0) + 1
    end
    # specialize the lengths carrying the most rows; ties broken by length for a stable build
    ranked = sort!(collect(keys(counts)); by = n -> (-counts[n], n))
    spec = Set(ranked[1:min(length(ranked), _MAX_SPECIALIZED_LENGTHS)])

    buckets = Any[]
    for L in sort!(collect(spec))
        rows = Int32[i for i in 1:D if lens[i] == L]
        isempty(rows) && continue
        nb = Matrix{Int32}(undef, L, length(rows))
        @inbounds for (k, i) in enumerate(rows)
            lo = g.starts[i]
            for w in 1:L
                nb[w, k] = g.flat[lo + w - 1]
            end
        end
        push!(buckets, _LengthBucket{T,L}(rows, nb, g.value))
    end

    trows = Int32[i for i in 1:D if !(lens[i] in spec)]
    tstarts = Vector{Int32}(undef, length(trows) + 1)
    tstarts[1] = 1
    @inbounds for (k, i) in enumerate(trows)
        tstarts[k + 1] = tstarts[k] + lens[i]
    end
    tflat = Vector{Int32}(undef, Int(tstarts[end]) - 1)
    @inbounds for (k, i) in enumerate(trows)
        lo = g.starts[i]
        for w in 1:lens[i]
            tflat[Int(tstarts[k]) + w - 1] = g.flat[lo + w - 1]
        end
    end

    return Tuple(buckets), _TailBucket{T}(trows, tstarts, tflat, g.value)
end

"""
    _bucket_apply!(Y, X, diags, bk::_LengthBucket, lo, hi, ::Val{bs})

Run one specialized bucket over its rows `lo:hi`. The body is generated with `L` unrolled gathers
and `_PAIR` rows in flight, so nothing about the trip count is decided at run time.
"""
@generated function _bucket_apply!(Y, X, diags, bk::_LengthBucket{T,L},
                                   lo::Int, hi::Int, ::Val{bs}) where {T,L,bs}
    pair = Expr(:block)
    for r in 0:_PAIR-1
        push!(pair.args, :($(Symbol(:i_, r)) = Int(@inbounds rows[k + $r])))
        push!(pair.args, :($(Symbol(:a_, r)) = @inbounds(diags[$(Symbol(:i_, r))]) *
                                               _loadv(X, $(Symbol(:i_, r)), Val(bs))))
        push!(pair.args, :($(Symbol(:s_, r)) = zero(SVector{bs,T})))
    end
    for w in 1:L, r in 0:_PAIR-1
        push!(pair.args, :($(Symbol(:s_, r)) += _loadv(X, Int(@inbounds nb[$w, k + $r]), Val(bs))))
    end
    for r in 0:_PAIR-1
        push!(pair.args, :(_storev!(Y, $(Symbol(:i_, r)),
                                    $(Symbol(:a_, r)) + value * $(Symbol(:s_, r)))))
    end

    single = Expr(:block,
        :(i = Int(@inbounds rows[k])),
        :(a = @inbounds(diags[i]) * _loadv(X, i, Val(bs))),
        :(sm = zero(SVector{bs,T})))
    for w in 1:L
        push!(single.args, :(sm += _loadv(X, Int(@inbounds nb[$w, k]), Val(bs))))
    end
    push!(single.args, :(_storev!(Y, i, a + value * sm)))

    quote
        rows = bk.rows; nb = bk.nb; value = bk.value
        k = lo
        while k + $(_PAIR) - 1 <= hi
            $pair
            k += $(_PAIR)
        end
        while k <= hi
            $single
            k += 1
        end
        return nothing
    end
end

"""
    _tail_apply!(Y, X, diags, bk::_TailBucket, lo, hi, ::Val{bs})

Run the unspecialized rows with a runtime trip count — the `row_read` inner loop, applied to the
minority of rows whose length did not earn a compiled body.
"""
function _tail_apply!(Y, X, diags::Vector{T}, bk::_TailBucket{T},
                      lo::Int, hi::Int, ::Val{bs}) where {T,bs}
    rows = bk.rows; starts = bk.starts; flat = bk.flat; value = bk.value
    @inbounds for k in lo:hi
        i = Int(rows[k])
        a = diags[i] * _loadv(X, i, Val(bs))
        sm = zero(SVector{bs,T})
        for c in Int(starts[k]):(Int(starts[k + 1]) - 1)
            sm += _loadv(X, Int(flat[c]), Val(bs))
        end
        _storev!(Y, i, a + value * sm)
    end
    return nothing
end

"""
    _row_bucketed!(Y, X, diags, buckets, tail, ::Val{bs})

Apply the operator by walking each bucket in turn. Buckets partition the rows, so every output row
is written exactly once and the buckets may be threaded independently.
"""
function _row_bucketed!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, diags::Vector{T},
                        buckets::Tuple, tail::_TailBucket{T}, ::Val{bs}) where {T,bs}
    # Serial fast path. Unlike `row_read`, which has ONE parallel region per apply, this kernel has
    # one per bucket — so on a single thread the `@threads` setup would be paid ~10 times for no
    # benefit. Measured: skipping it took bs = 1 from 0.114 ms back to 0.045 ms (allocation from
    # 7632 B to 0), which is the whole advantage this kernel exists for.
    if Threads.nthreads() == 1
        for bk in buckets
            _bucket_apply!(Y, X, diags, bk, 1, length(bk.rows), Val(bs))
        end
        _tail_apply!(Y, X, diags, tail, 1, length(tail.rows), Val(bs))
        return nothing
    end

    for bk in buckets
        n = length(bk.rows)
        n == 0 && continue
        nt = max(1, min(Threads.nthreads(), cld(n, 64)))
        chunk = cld(n, nt)
        Threads.@threads :static for t in 1:nt
            _bucket_apply!(Y, X, diags, bk, (t - 1) * chunk + 1, min(t * chunk, n), Val(bs))
        end
    end
    n = length(tail.rows)
    if n > 0
        nt = max(1, min(Threads.nthreads(), cld(n, 64)))
        chunk = cld(n, nt)
        Threads.@threads :static for t in 1:nt
            _tail_apply!(Y, X, diags, tail, (t - 1) * chunk + 1, min(t * chunk, n), Val(bs))
        end
    end
    return nothing
end

"""
    _build_row_bucketed(s::MappingStructure, ::Val{bs}) -> Function

Close over the bucketed structure and return the row-layout matvec `(Y, X) -> nothing`.

Bucketing happens here, once at setup. Takes no tuning parameters.
"""
function _build_row_bucketed(s::MappingStructure{T,TI}, ::Val{bs}) where {T,TI,bs}
    buckets, tail = _build_buckets(s)
    diags = s.diags
    return (Y, X) -> _row_bucketed!(Y, X, diags, buckets, tail, Val(bs))
end
