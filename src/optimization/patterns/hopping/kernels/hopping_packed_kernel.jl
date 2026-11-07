# hopping / packed — the default kernel: SIMD with the axis-1 neighbours taken from registers.
#
# Pack `P = W / bs` consecutive SITES of one x-line into a `Vec{W,T}`: lane `w` (0-based) holds
# block `w % bs` of site offset `w ÷ bs`. Because a pack is P-aligned inside the line, the axis-1
# neighbours of a whole pack are the same P sites shifted by one, which straddles the register
# boundary — so they are recovered from the ADJACENT packs with two fixed shuffles:
#
#   xm = shufflevector(prev, cur, minus)     lane w ← site (w÷bs) - 1, same block
#   xp = shufflevector(cur, next, plus)      lane w ← site (w÷bs) + 1, same block
#
# and `cur`/`next` are carried into the following step, so each pack is LOADED ONCE and used as
# `next`, then `cur`, then `prev`. The 2D+1-point stencil therefore has 2D-1 input streams, not
# 2D+1. Both permutations are compile-time immediates.
#
# At P = 1 (W = bs) the permutations degenerate to "take all of prev" and "take all of next" and
# the shuffles vanish entirely — the kernel becomes pure register rotation. That is not a corner
# case to avoid: it is the correct code for a lattice whose x-extent does not admit a wider pack,
# and it is why `_build_hop_packed` can always fall back to a legal `W`.
#
# Measured on THIS implementation (EPYC 9354, 16 threads, Float64, default parameters, ms per
# vector; `rows` alongside for the ratio):
#
#     lattice          bs=1              bs=4              bs=8
#     128^3        0.132 / 0.355     0.094 / 0.191     0.165 / 0.209
#     2048x1024    0.103 / 0.256     0.076 / 0.138     0.155 / 0.240
#     12^6         0.780 / 1.719     0.298 / 0.700     0.499 / 0.617
#     2^21 (1-D)   0.062 / 0.159     0.057 / 0.166     0.141 / 0.129
#
# The gap is widest exactly where `rows` has no SIMD to work with: at bs = 1 an `SVector{1}` is a
# scalar, while a pack still holds P = W whole sites. It closes as bs grows into a full register
# on its own — at bs = 8 in one dimension `rows` is already the faster of the two.
#
# CORRECTNESS PRECONDITIONS, not optimisations:
#   * `bs <= W` and `W % bs == 0`. Lane w holds site offset w÷bs and block w%bs; the shift by one
#     site is the lane permutation `w -> w ± bs`, which is only a permutation of the pack when bs
#     divides W.
#   * `P` must divide `L₁`, or a pack would straddle two x-lines and the shuffled neighbours
#     would come from the wrong line. `_build_hop_packed` halves `W` until it does.
#   * `T <: Real` — SIMD.jl `Vec` carries no `Complex`.
#   * `Y`/`X` must be dense `Matrix`: the kernel takes raw pointers.

"""
    HOP_PACKED_PARAMS

This kernel's tunable parameters and their defaults.

- `W`: pack width in elements. `0` means **derive from the hardware** (see
  [`_hop_lane_width`](@ref)); any other value pins it. It is then reduced, if necessary, until
  `P = W/bs` divides the x-extent — a correctness repair, not a preference.
- `tile`: x-lines per parallel work chunk. `0` gives each thread one contiguous range, which has
  the fewest chunk boundaries and the best reuse of the orthogonal neighbour streams.

```julia
MappingKernel(:hopping, :packed)                      # W derived, one chunk per thread
MappingKernel(:hopping, :packed; W = 8, tile = 64)    # pinned, e.g. for an AVX2 host
```

`W` is not a free choice on this kernel the way it is on a gather kernel: it sets `P`, and `P`
sets how much of the x-line one register covers. The derived width is a good default but not a
universal optimum — swept on `128³` at 16 threads (ms per vector):

| | `W=4` | `W=8` | `W=16` | `W=32` |
|:--|--:|--:|--:|--:|
| `bs = 1` | 0.235 | 0.142 | **0.107** | 0.187 |
| `bs = 8` | — | **0.175** | 0.216 | 0.178 |

At `bs = 1` the widest NATIVE pack wins outright and going past it (`W = 32`, two registers)
loses; at `bs = 8` the narrower pack wins instead, because `P = W/bs` is already small and the
wider shuffles stop paying for themselves. The optimum moves with `bs`, with the ISA, with
`sizeof(T)` and with `D` — sweep it with [`benchmark_mapping`](@ref) rather than assuming.
"""
const HOP_PACKED_PARAMS = (W = 0, tile = 0)

"""
    _hop_lane_width(::Type{T}) -> Int

The pack width in elements: 16 `Float64` lanes where AVX-512 is present, 8 otherwise (and double
that for 32-bit element types).

Detected at BUILD time rather than baked into a `const`, which would freeze whatever the machine
that ran precompilation happened to support into the cached image. Unknown architectures fall
back to the narrow width, which is always correct — `W` only affects speed.

`spin_flip/packed` carries its own copy of this deliberately: a pattern folder is self-contained,
so neither reaches into the other.
"""
function _hop_lane_width(::Type{T}) where {T}
    wide = false
    if Sys.ARCH === :x86_64
        wide = try
            occursin("avx512f", read("/proc/cpuinfo", String))
        catch
            false
        end
    end
    base = wide ? 16 : 8
    return sizeof(T) <= 4 ? 2base : base
end

"""
    _hop_diag_pack(pd, site, ::Val{W}, ::Val{bs}) -> Vec{W,T}

The on-site energies of one pack, in lane order: lane `w` gets `diags[site + w÷bs]`. At `bs = 1`
that is a plain contiguous load; above it the `P` distinct values are broadcast into their lanes
without ever materialising a `bs`-times-expanded copy of the diagonal.
"""
@generated function _hop_diag_pack(pd::Ptr{T}, site::Int, ::Val{W}, ::Val{bs}) where {T,W,bs}
    bs == 1 && return :(vload(Vec{$W,T}, pd + _hop_byteoffset(T, site)))
    vals = [:(unsafe_load(pd, site + $(fld(w - 1, bs)) + 1)) for w in 1:W]
    return :(Vec{$W,T}($(Expr(:tuple, vals...))))
end

"""
    _hop_packed_line!(pY, pX, pd, base, nbase, plo, phi, npacks, hopping, ::Val{W}, ::Val{bs})

Slide packs `plo : phi-1` of the x-line starting at 0-based site `base`, whose `2(D-1)`
orthogonal neighbour line bases are `nbase`.

Generated: the axis-1 permutations are immediates, and the orthogonal loads are either fully
unrolled (`D <= MAX_UNROLLED_DIM`) or a counted loop over `nbase`.

The final pack of a LINE is peeled out of the hot loop rather than branching inside it: only
there does `next` wrap back to pack 0. A range that stops short of the line end never reaches
the peel, which is what makes a split line as cheap as a whole one.
"""
@generated function _hop_packed_line!(pY::Ptr{T}, pX::Ptr{T}, pd::Ptr{T}, base::Int,
                                      nbase::NTuple{N,Int}, plo::Int, phi::Int, npacks::Int,
                                      hopping::T, ::Val{W}, ::Val{bs}) where {T,N,W,bs}
    P = W ÷ bs
    minus = ntuple(i -> (l = i - 1; l < bs ? W - bs + l : W + l - bs), W)
    plus  = ntuple(i -> (l = i - 1; l < W - bs ? l + bs : W + l - (W - bs)), W)

    nsum = if N == 0
        :(xm + xp)
    elseif _hop_unrolled(N)
        orth = _hop_pairwise_add(Any[
            :(vload(Vec{$W,T}, pX + _hop_byteoffset(T, (nbase[$j] + x) * $bs))) for j in 1:N])
        :((xm + xp) + $orth)
    else
        quote
            o = zero(Vec{$W,T})
            for j in 1:$N
                o += vload(Vec{$W,T}, pX + _hop_byteoffset(T, (nbase[j] + x) * $bs))
            end
            (xm + xp) + o
        end
    end

    body(nextexpr) = quote
        x = p * $P
        site = base + x
        next = $nextexpr
        xm = shufflevector(prev, cur, Val($minus))
        xp = shufflevector(cur, next, Val($plus))
        a = muladd(h, $nsum, _hop_diag_pack(pd, site, Val($W), Val($bs)) * cur)
        vstore(a, pY + _hop_byteoffset(T, site * $bs))
        prev = cur
        cur = next
        p += 1
    end

    return quote
        @inbounds begin
            h = Vec{$W,T}(hopping)
            prevpack = plo == 0 ? npacks - 1 : plo - 1
            prev = vload(Vec{$W,T}, pX + _hop_byteoffset(T, (base + prevpack * $P) * $bs))
            cur  = vload(Vec{$W,T}, pX + _hop_byteoffset(T, (base + plo * $P) * $bs))
            p = plo
            interior = min(phi, npacks - 1)
            while p < interior
                $(body(:(vload(Vec{$W,T},
                               pX + _hop_byteoffset(T, (base + (p + 1) * $P) * $bs)))))
            end
            while p < phi                          # the line's last pack: `next` wraps to pack 0
                $(body(:(vload(Vec{$W,T}, pX + _hop_byteoffset(T, base * $bs)))))
            end
        end
        return nothing
    end
end

"""
    _hop_packed_units!(pY, pX, pd, shape, strides, hopping, u0, u1, xsplit, npacks, …)

Run work units `u0 : u1-1`. Decoding a unit into (line, pack range) and hoisting the line's
orthogonal bases is per-line work; everything per-site lives in [`_hop_packed_line!`](@ref).
"""
function _hop_packed_units!(pY::Ptr{T}, pX::Ptr{T}, pd::Ptr{T}, shape::NTuple{D,Int},
                            strides::NTuple{D,Int}, hopping::T, u0::Int, u1::Int,
                            xsplit::Int, npacks::Int, ::Val{W}, ::Val{bs}) where {T,D,W,bs}
    for u in u0:(u1 - 1)
        line, plo, phi = _hop_unit_range(u, xsplit, npacks)
        phi > plo || continue
        nbase = _hop_line_bases(shape, strides, line)
        _hop_packed_line!(pY, pX, pd, line * shape[1], nbase, plo, phi, npacks,
                          hopping, Val(W), Val(bs))
    end
    return nothing
end

"""
    _hop_packed_apply!(Y, X, diags, shape, strides, hopping, ::Val{W}, ::Val{bs}, tile)

Apply the operator over the whole lattice. `Y`/`X` must be dense `Matrix` — the kernel takes raw
pointers, so a view or a lazy transpose is not acceptable here.

One parallel region per apply, chunked over work units and never indexed by `threadid()` (tasks
migrate on Julia >= 1.12). The single-thread path skips the region entirely so the kernel stays
allocation-free when there is nothing to parallelise.
"""
function _hop_packed_apply!(Y::Matrix{T}, X::Matrix{T}, diags::Vector{T}, shape::NTuple{D,Int},
                            strides::NTuple{D,Int}, hopping::T, ::Val{W}, ::Val{bs},
                            tile::Int) where {T,D,W,bs}
    L1 = shape[1]
    npacks = L1 ÷ (W ÷ bs)
    nlines = length(diags) ÷ L1
    nt = Threads.nthreads()
    xsplit = _hop_xsplit(nlines, npacks, nt)
    nunits = nlines * xsplit
    chunk, nchunks = _hop_chunking(nunits, xsplit, tile, nt)
    GC.@preserve Y X diags begin
        pY, pX, pd = pointer(Y), pointer(X), pointer(diags)
        if nt == 1 || nchunks == 1
            _hop_packed_units!(pY, pX, pd, shape, strides, hopping, 0, nunits,
                               xsplit, npacks, Val(W), Val(bs))
        else
            Threads.@threads :static for c in 1:nchunks
                _hop_packed_units!(pY, pX, pd, shape, strides, hopping,
                                   (c - 1) * chunk, min(c * chunk, nunits),
                                   xsplit, npacks, Val(W), Val(bs))
            end
        end
    end
    return nothing
end

"""
    _build_hop_packed(hs::HopStructure, ::Val{bs}; W, tile) -> Function

Close over the lattice and return the row-layout matvec `(Y, X) -> nothing`, with the pack width
and the dimension lifted into the type domain.

`W = 0` derives the pack width from the hardware; anything else pins it. It is then HALVED until
`P = W/bs` divides the x-extent, which always terminates: `P = 1` divides everything, and at
`P = 1` the kernel is exactly the register-rotation form. A pinned `W` that has to be reduced is
therefore honoured as an upper bound, not rejected.

# Throws
- `ArgumentError` for a complex element type, a non-power-of-two `W`, `bs > W`, `W % bs != 0`, or
  a negative `tile` — the first four are correctness preconditions of the lane permutation, not
  preferences.
"""
function _build_hop_packed(hs::HopStructure{T,D}, ::Val{bs}; W::Int = HOP_PACKED_PARAMS.W,
                           tile::Int = HOP_PACKED_PARAMS.tile) where {T,D,bs}
    T <: Real || throw(ArgumentError(
        "hopping/packed: is built on SIMD Vec, which carries no Complex. Use hopping/rows."))
    W = W == 0 ? _hop_lane_width(T) : W
    (W > 0 && ispow2(W)) || throw(ArgumentError(
        "hopping/packed: W must be a positive power of two (0 = derive), got $W."))
    bs <= W || throw(ArgumentError(
        "hopping/packed: bs = $bs exceeds the pack width W = $W. Use hopping/rows."))
    rem(W, bs) == 0 || throw(ArgumentError(
        "hopping/packed: bs = $bs must divide the pack width W = $W (the one-site shift is only " *
        "a lane permutation then). Use hopping/rows."))
    tile >= 0 || throw(ArgumentError("hopping/packed: tile must be >= 0, got $tile."))

    L1 = hs.shape[1]
    while W > bs && rem(L1, W ÷ bs) != 0        # a pack must not straddle two x-lines
        W >>= 1
    end

    # Every `Val` is built HERE, not in the closure body: `W` is an ordinary `Int` at this point,
    # so `Val(W)` inside the closure would be constructed from a runtime value on every apply —
    # a dynamic dispatch on the hot path.
    diags, shape, strides, hopping = hs.diags, hs.shape, hs.strides, hs.hopping
    Wv = Val(W)
    return (Y, X) -> _hop_packed_apply!(Y, X, diags, shape, strides, hopping, Wv, Val(bs), tile)
end

"""
    _hop_packed_units(hs::HopStructure, ::Val{bs}; W) -> Int

How many parallelizable work units this kernel has: `nlines × xsplit`. A unit is a slice of one
lattice line, NOT a basis state. `W` is reduced exactly as in [`_build_hop_packed`](@ref) so the
count matches what the range form will actually iterate.
"""
function _hop_packed_units(hs::HopStructure{T,D}, ::Val{bs};
                           W::Int = HOP_PACKED_PARAMS.W) where {T,D,bs}
    W = W == 0 ? _hop_lane_width(T) : W
    L1 = hs.shape[1]
    while W > bs && rem(L1, W ÷ bs) != 0
        W >>= 1
    end
    npacks = L1 ÷ (W ÷ bs)
    nlines = length(hs.diags) ÷ L1
    return nlines * _hop_xsplit(nlines, npacks, Threads.nthreads())
end

"""
    _build_hop_packed_range(hs::HopStructure, ::Val{bs}; W) -> Function

The SERIAL range form `(Y, X, lo, hi) -> nothing` over WORK UNITS `lo:hi` (1-based, inclusive;
see [`_hop_packed_units`](@ref)), for callers that own the threads. `Y`/`X` must be dense
`Matrix` — the kernel takes raw pointers.
"""
function _build_hop_packed_range(hs::HopStructure{T,D}, ::Val{bs};
                                 W::Int = HOP_PACKED_PARAMS.W) where {T,D,bs}
    T <: Real || throw(ArgumentError(
        "hopping/packed: is built on SIMD Vec, which carries no Complex. Use hopping/rows."))
    W = W == 0 ? _hop_lane_width(T) : W
    L1 = hs.shape[1]
    while W > bs && rem(L1, W ÷ bs) != 0
        W >>= 1
    end
    npacks = L1 ÷ (W ÷ bs)
    nlines = length(hs.diags) ÷ L1
    xsplit = _hop_xsplit(nlines, npacks, Threads.nthreads())
    diags, shape, strides, hopping = hs.diags, hs.shape, hs.strides, hs.hopping
    Wv = Val(W)
    return (Y::Matrix{T}, X::Matrix{T}, lo::Int, hi::Int) -> begin
        GC.@preserve Y X diags begin
            _hop_packed_units!(pointer(Y), pointer(X), pointer(diags), shape, strides, hopping,
                               lo - 1, hi, xsplit, npacks, Wv, Val(bs))
        end
        return nothing
    end
end
