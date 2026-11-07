# hopping / rows — the general kernel: one site's whole block in an `SVector{bs}`, slid along x.
#
# Same skeleton as `packed`, one difference: a register holds ALL bs blocks of ONE site rather
# than P sites of one block. An axis-1 neighbour is then a whole register, so the slide needs no
# shuffle at all —
#
#   xm = prev        xp = next
#
# — and each site is still loaded once for all three roles. This is the `P = 1` geometry, reached
# here without any of `packed`'s preconditions: `bs` need not divide a pack width, the x-extent
# need not divide anything, the element type may be `Complex`, and `Y`/`X` may be any
# `AbstractMatrix` because nothing is addressed through a raw pointer.
#
# It is the fallback in the same sense `spin_flip/rows` is: everything `packed` refuses lands
# here and is still correct. It is not merely a safety net, though — at large `bs` the SVector is
# already a full vector register or several, so the two converge and can cross (EPYC 9354, 16
# threads, Float64, ms per vector, packed / rows):
#
#     lattice          bs=1              bs=4              bs=8
#     128^3        0.132 / 0.355     0.094 / 0.191     0.165 / 0.209
#     2^21 (1-D)   0.062 / 0.159     0.057 / 0.166     0.141 / 0.129     <- rows wins
#
# Where it genuinely loses is bs = 1, where `SVector{1}` is a scalar and there is no SIMD left to
# do. `packed` owns that regime, which is why `auto_score` puts this one below it.

"""
    HOP_ROWS_PARAMS

This kernel's tunable parameter and its default.

- `tile`: x-lines per parallel work chunk. `0` gives each thread one contiguous range, which has
  the fewest chunk boundaries and the best reuse of the orthogonal neighbour streams. A small
  positive value trades that locality for load balance.

```julia
MappingKernel(:hopping, :rows)                # one chunk per thread
MappingKernel(:hopping, :rows; tile = 32)     # 32 x-lines per chunk
```

There is deliberately no `W` here: the register width IS `bs`, chosen by the Krylov block size
rather than by this kernel.
"""
const HOP_ROWS_PARAMS = (tile = 0,)

"""    _hop_loadv(X, site, ::Val{bs}, ::Type{T}) -> SVector{bs,T} — the block at 0-based `site`."""
@inline _hop_loadv(X::AbstractMatrix, site::Int, ::Val{bs}, ::Type{T}) where {bs,T} =
    SVector{bs,T}(ntuple(b -> @inbounds(X[b, site + 1]), Val(bs)))

"""    _hop_storev!(Y, site, v) — write the `bs` amplitudes of 0-based `site` back."""
@inline function _hop_storev!(Y::AbstractMatrix, site::Int, v::SVector{bs}) where {bs}
    @inbounds for b in 1:bs
        Y[b, site + 1] = v[b]
    end
    return nothing
end

"""
    _hop_rows_line!(Y, X, diags, base, nbase, L1, xlo, xhi, hopping, ::Val{bs})

Slide sites `xlo : xhi-1` of the x-line starting at 0-based site `base`, whose `2(D-1)`
orthogonal neighbour line bases are `nbase`.

Generated: the orthogonal loads are either fully unrolled (`D <= MAX_UNROLLED_DIM`) or a counted
loop over `nbase`. The line's last site is peeled out of the hot loop — only there does `next`
wrap back to site 0 of the line.
"""
@generated function _hop_rows_line!(Y, X, diags::Vector{T}, base::Int, nbase::NTuple{N,Int},
                                    L1::Int, xlo::Int, xhi::Int, hopping::T,
                                    ::Val{bs}) where {T,N,bs}
    nsum = if N == 0
        :(prev + next)
    elseif _hop_unrolled(N)
        orth = _hop_pairwise_add(Any[
            :(_hop_loadv(X, nbase[$j] + x, Val($bs), T)) for j in 1:N])
        :((prev + next) + $orth)
    else
        quote
            o = zero(SVector{$bs,T})
            for j in 1:$N
                o += _hop_loadv(X, nbase[j] + x, Val($bs), T)
            end
            (prev + next) + o
        end
    end

    body(nextexpr) = quote
        site = base + x
        next = $nextexpr
        _hop_storev!(Y, site, muladd(hopping, $nsum, diags[site + 1] * cur))
        prev = cur
        cur = next
        x += 1
    end

    return quote
        @inbounds begin
            prev = _hop_loadv(X, base + (xlo == 0 ? L1 - 1 : xlo - 1), Val($bs), T)
            cur  = _hop_loadv(X, base + xlo, Val($bs), T)
            x = xlo
            interior = min(xhi, L1 - 1)
            while x < interior
                $(body(:(_hop_loadv(X, base + x + 1, Val($bs), T))))
            end
            while x < xhi                          # the line's last site: `next` wraps to site 0
                $(body(:(_hop_loadv(X, base, Val($bs), T))))
            end
        end
        return nothing
    end
end

"""
    _hop_rows_units!(Y, X, diags, shape, strides, hopping, u0, u1, xsplit, ::Val{bs})

Run work units `u0 : u1-1`. Decoding a unit into (line, site range) and hoisting the line's
orthogonal bases is per-line work; everything per-site lives in [`_hop_rows_line!`](@ref).
"""
function _hop_rows_units!(Y, X, diags::Vector{T}, shape::NTuple{D,Int}, strides::NTuple{D,Int},
                          hopping::T, u0::Int, u1::Int, xsplit::Int, ::Val{bs}) where {T,D,bs}
    L1 = shape[1]
    for u in u0:(u1 - 1)
        line, xlo, xhi = _hop_unit_range(u, xsplit, L1)
        xhi > xlo || continue
        nbase = _hop_line_bases(shape, strides, line)
        _hop_rows_line!(Y, X, diags, line * L1, nbase, L1, xlo, xhi, hopping, Val(bs))
    end
    return nothing
end

"""
    _hop_rows_apply!(Y, X, hs::HopStructure, ::Val{bs}, tile)

Apply the operator over the whole lattice, for any `AbstractMatrix` arguments.

One parallel region per apply, chunked over work units and never indexed by `threadid()` (tasks
migrate on Julia >= 1.12). The single-thread path skips the region entirely so the kernel stays
allocation-free when there is nothing to parallelise.
"""
function _hop_rows_apply!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, hs::HopStructure{T,D},
                          ::Val{bs}, tile::Int) where {T,D,bs}
    shape, strides = hs.shape, hs.strides
    L1 = shape[1]
    nlines = hs.dim ÷ L1
    nt = Threads.nthreads()
    xsplit = _hop_xsplit(nlines, L1, nt)
    nunits = nlines * xsplit
    chunk, nchunks = _hop_chunking(nunits, xsplit, tile, nt)
    if nt == 1 || nchunks == 1
        _hop_rows_units!(Y, X, hs.diags, shape, strides, hs.hopping, 0, nunits, xsplit, Val(bs))
        return nothing
    end
    Threads.@threads :static for c in 1:nchunks
        _hop_rows_units!(Y, X, hs.diags, shape, strides, hs.hopping,
                         (c - 1) * chunk, min(c * chunk, nunits), xsplit, Val(bs))
    end
    return nothing
end

"""
    _build_hop_rows(hs::HopStructure, ::Val{bs}; tile) -> Function

Close over the lattice and return the row-layout matvec `(Y, X) -> nothing`. Accepts any `bs`,
any element type, and any `AbstractMatrix` argument.

# Throws
- `ArgumentError` if `tile` is negative.
"""
function _build_hop_rows(hs::HopStructure{T,D}, ::Val{bs};
                         tile::Int = HOP_ROWS_PARAMS.tile) where {T,D,bs}
    tile >= 0 || throw(ArgumentError("hopping/rows: tile must be >= 0, got $tile."))
    return (Y, X) -> _hop_rows_apply!(Y, X, hs, Val(bs), tile)
end

"""
    _hop_rows_units(hs::HopStructure, ::Val{bs}) -> Int

How many parallelizable work units this kernel has: `nlines × xsplit`. A unit is a slice of one
lattice line, NOT a basis state — the hopping kernels partition the lattice, so this is the axis
a caller that owns the threads must split. See [`build_range_kernel`](@ref).
"""
function _hop_rows_units(hs::HopStructure{T,D}, ::Val{bs}) where {T,D,bs}
    L1 = hs.shape[1]
    nlines = hs.dim ÷ L1
    return nlines * _hop_xsplit(nlines, L1, Threads.nthreads())
end

"""
    _build_hop_rows_range(hs::HopStructure, ::Val{bs}) -> Function

The SERIAL range form `(Y, X, lo, hi) -> nothing` over WORK UNITS `lo:hi` (1-based, inclusive;
see [`_hop_rows_units`](@ref) for the count), for callers that own the threads.
"""
function _build_hop_rows_range(hs::HopStructure{T,D}, ::Val{bs}) where {T,D,bs}
    shape, strides = hs.shape, hs.strides
    L1 = shape[1]
    nlines = hs.dim ÷ L1
    xsplit = _hop_xsplit(nlines, L1, Threads.nthreads())
    diags, hopping = hs.diags, hs.hopping
    return (Y, X, lo::Int, hi::Int) ->
        _hop_rows_units!(Y, X, diags, shape, strides, hopping, lo - 1, hi, xsplit, Val(bs))
end
