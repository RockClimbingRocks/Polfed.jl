# spin_flip / packed — the default kernel: column-packed SIMD with in-register low-bit flips.
#
# Pack `P = W / bs` consecutive Hilbert columns into one `Vec{W,T}`. Because a pack base is
# P-aligned, the P packed columns differ ONLY in the low log2(P) bits of the state index, so
# every mask splits at that boundary as `mask = mhi | mlo`, `mlo = mask & (P-1)`:
#
#   mlo  -> a lane permutation INSIDE the register: one shuffle, zero memory traffic
#   mhi  -> the whole pack moves to another P-aligned pack, and the xor offset is identical
#           for all P columns: ONE full-width load serves the whole pack
#   mhi == 0 -> the source pack is already in a register: no load at all
#
# With the masks known at compile time every shuffle constant and load offset is an immediate.
# `G` packs are processed with independent accumulators.
#
# Measured defaults (QSun, L=20, t=1, znver4, per-column ms):
#
#              bs1     bs2     bs4     bs8
#   W=16 G=4   1.263   1.466   1.544   1.514      <- the defaults
#   W=16 G=2   1.312   1.710   1.708   1.808
#   W=16 G=1   1.443   2.612   1.994   2.348
#   W=8  G=4   1.336   1.770   1.738   1.833
#
# Both are EXPOSED as tunable parameters. The table above is one machine, one element type and
# one model: W trades vector width against register pressure and G trades ILP against the same
# registers, so the optimum moves with the ISA (AVX2 vs AVX-512), with `bs` (which sets
# P = W/bs), with sizeof(T), and with the number of terms M. Defaults that measured well are
# not the same thing as constants — sweep them per machine with `benchmark_mapping`.
#
# CORRECTNESS PRECONDITIONS, not optimisations:
#   * `bs` must be a power of two and `<= W`. Lane w holds column offset w/bs and block index
#     w%bs; flipping the column offset by mlo maps w -> w xor (mlo*bs), which is a permutation
#     ONLY because bs is a power of two (so mlo*bs has no bits below bs).
#   * a pack base must be a multiple of P, else the P columns do not differ only in low bits
#     and every shuffle is wrong. Thread chunks are therefore aligned to P*G.
#   * `xor(p0, mhi)` preserves P-alignment precisely because mhi has no bits below P.

"""
    PACKED_PARAMS

This kernel's tunable parameters and their measured defaults.

- `W`: pack width in elements. `0` means **derive from the hardware** (see
  [`_packed_lane_width`](@ref)); any other value pins it. Must be a power of two and at least
  `bs`.
- `G`: packs accumulated in flight, for instruction-level parallelism.

```julia
MappingKernel(:spin_flip, :packed)                  # W derived, G = 4
MappingKernel(:spin_flip, :packed; W = 8, G = 2)    # pinned, e.g. for an AVX2 host
```

Total live accumulator state is roughly `W · G` elements, so the two trade against each other
against the register file — do not tune one without the other.
"""
const PACKED_PARAMS = (W = 0, G = 4)

"""
    _packed_lane_width(::Type{T}) -> Int

The pack width in elements: 16 `Float64` lanes where AVX-512 is present, 8 otherwise (and
double that for 32-bit element types). Derived, never asked of the caller.

Detected at BUILD time rather than baked into a `const`: a `const` would be evaluated during
precompilation and would freeze whatever the build machine happened to support into the cached
image. Unknown architectures fall back to the narrow width, which is always correct — `W` only
affects speed.
"""
function _packed_lane_width(::Type{T}) where {T}
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
    _packed_block!(pY, pX, pd, values, p0, ::Val{MASKS}, ::Val{W}, ::Val{bs}, ::Val{G})

Compute `G` packs starting at 0-based, P-aligned column `p0`, straight through raw pointers.

Generated: with `MASKS` a compile-time tuple, each term becomes either a bare register
reference, one `shufflevector` with an immediate permutation, or one `vload` at an immediate
offset — no mask loop, no index arithmetic at run time.
"""
@generated function _packed_block!(pY::Ptr{T}, pX::Ptr{T}, pd::Ptr{T}, values::SVector{M,T},
                                   p0::Int, ::Val{MASKS}, ::Val{W}, ::Val{bs},
                                   ::Val{G}) where {T,M,MASKS,W,bs,G}
    P  = W ÷ bs
    sz = sizeof(T)
    init, upd, store = Expr[], Expr[], Expr[]
    for g in 0:G-1
        v, a, dv = Symbol(:v_, g), Symbol(:a_, g), Symbol(:d_, g)
        eoff, coff = g * W, g * P
        push!(init, :($v = vload(Vec{$W,T}, pX + (p0 * $bs + $eoff) * $sz)))
        if bs == 1
            push!(init, :($dv = vload(Vec{$W,T}, pd + (p0 + $coff) * $sz)))
        else
            # lane w (1-based) belongs to column p0 + coff + (w-1)/bs
            push!(init, :($dv = Vec{$W,T}($(Expr(:tuple,
                [:(unsafe_load(pd, p0 + $(coff + (w - 1) ÷ bs) + 1)) for w in 1:W]...)))))
        end
        push!(init, :($a = $dv * $v))
        push!(store, :(vstore($a, pY + (p0 * $bs + $eoff) * $sz)))
    end
    for k in 1:M
        mask = MASKS[k]
        mlo, mhi = mask & (P - 1), mask & ~(P - 1)
        perm = ntuple(w -> xor(w - 1, mlo * bs), W)          # 0-based lane permutation
        for g in 0:G-1
            v, a, coff = Symbol(:v_, g), Symbol(:a_, g), g * P
            src = if mhi == 0
                mlo == 0 ? v : :(shufflevector($v, Val($perm)))
            else
                ld = :(vload(Vec{$W,T}, pX + (xor(p0 + $coff, $mhi) * $bs) * $sz))
                mlo == 0 ? ld : :(shufflevector($ld, Val($perm)))
            end
            push!(upd, :($a = muladd(Vec{$W,T}(@inbounds(values[$k])), $src, $a)))
        end
    end
    quote
        $(init...)
        $(upd...)
        $(store...)
        return nothing
    end
end

"""
    _packed_apply!(Y, X, diags, values, ::Val{MASKS}, ::Val{W}, ::Val{bs}, ::Val{G})

Apply the operator over the whole space. `Y`/`X` must be dense `Matrix` — the kernel takes raw
pointers, so a view or a lazy transpose is not acceptable here.
"""
function _packed_apply!(Y::Matrix{T}, X::Matrix{T}, diags::Vector{T}, values::SVector{M,T},
                        ::Val{MASKS}, ::Val{W}, ::Val{bs}, ::Val{G}) where {T,M,MASKS,W,bs,G}
    D = size(X, 2)
    P = W ÷ bs
    align = P * G
    nt = Threads.nthreads() == 1 ? 1 : max(1, min(Threads.nthreads(), fld(D, align)))
    chunk = align * cld(cld(D, nt), align)          # every chunk start a multiple of P*G
    GC.@preserve Y X diags begin
        pY, pX, pd = pointer(Y), pointer(X), pointer(diags)
        if nt == 1
            _packed_range!(pY, pX, pd, values, 0, D, Val(MASKS), Val(W), Val(bs), Val(G))
        else
            Threads.@threads :static for t in 1:nt
                _packed_range!(pY, pX, pd, values, (t - 1) * chunk, min(t * chunk, D),
                               Val(MASKS), Val(W), Val(bs), Val(G))
            end
        end
    end
    return nothing
end

"""
    _packed_range!(pY, pX, pd, values, p0, hi, …)

Packs `p0 : hi` (0-based, exclusive `hi`): `G` packs at a time, then the single-pack tail.
Both loops are required — a dropped tail silently leaves stale columns in `Y`.
"""
function _packed_range!(pY::Ptr{T}, pX::Ptr{T}, pd::Ptr{T}, values::SVector{M,T},
                        p0::Int, hi::Int, ::Val{MASKS}, ::Val{W}, ::Val{bs},
                        ::Val{G}) where {T,M,MASKS,W,bs,G}
    P = W ÷ bs
    while p0 + G * P <= hi
        _packed_block!(pY, pX, pd, values, p0, Val(MASKS), Val(W), Val(bs), Val(G))
        p0 += G * P
    end
    while p0 + P <= hi
        _packed_block!(pY, pX, pd, values, p0, Val(MASKS), Val(W), Val(bs), Val(1))
        p0 += P
    end
    return nothing
end

"""
    _build_packed(fs::FlipStructure, ::Val{bs}) -> Function

Close over the flip structure and return the row-layout matvec `(Y, X) -> nothing`, with the
masks lifted into the type domain so the whole term sum is compiled to immediates.

`W = 0` derives the pack width from the hardware; anything else pins it. `G` is reduced when
the space is too small to hold `G` packs (`D < P·G`), which only happens at the tiny `L` used
in tests.

# Throws
- `ArgumentError` if `bs` is not a power of two, if `bs > W`, if `W` is not a power of two, or
  if `D` is smaller than one pack — these are correctness preconditions of the lane
  permutation, not preferences.
"""
function _build_packed(fs::FlipStructure{T}, ::Val{bs};
                       W::Int = PACKED_PARAMS.W, G::Int = PACKED_PARAMS.G) where {T,bs}
    W = W == 0 ? _packed_lane_width(T) : W
    (W > 0 && ispow2(W)) || throw(ArgumentError(
        "spin_flip/packed: W must be a positive power of two (0 = derive), got $W."))
    G >= 1 || throw(ArgumentError("spin_flip/packed: G must be >= 1, got $G."))
    ispow2(bs) || throw(ArgumentError(
        "spin_flip/packed: bs must be a power of two (the lane permutation is only a " *
        "permutation then), got $bs. Use spin_flip/rows."))
    bs <= W || throw(ArgumentError(
        "spin_flip/packed: bs = $bs exceeds the pack width W = $W. Use spin_flip/rows."))
    P = W ÷ bs
    fs.dim >= P || throw(ArgumentError(
        "spin_flip/packed: dim = $(fs.dim) is smaller than one pack (P = $P)."))
    # shrink G rather than fail when the space cannot hold G packs (small L in tests)
    while G > 1 && fs.dim < P * G
        G >>= 1
    end
    # Every `Val` is built HERE, not in the closure body. `W` and `G` are ordinary `Int`s at
    # this point, so `Val(W)` inside the closure would be constructed from a runtime value on
    # every apply — a dynamic dispatch that allocated 80 B per matvec before this was hoisted.
    masks = Val(Tuple(fs.masks))
    values = SVector{length(fs.values),T}(fs.values)
    diags = fs.diags
    Wv, Gv = Val(W), Val(G)
    return (Y, X) -> _packed_apply!(Y, X, diags, values, masks, Wv, Val(bs), Gv)
end

"""
    _build_packed_range(fs::FlipStructure, ::Val{bs}; W, G) -> Function

The SERIAL range form `(Y, X, lo, hi) -> nothing` over basis states `lo:hi` (1-based,
inclusive), for callers that own the threads.

Unlike the row kernels, `lo - 1` MUST be a multiple of `P·G = (W÷bs)·G`: the kernel steps in
whole packs and the lane permutation is only valid on a pack boundary. Query it with
[`range_alignment`](@ref) and chunk accordingly — the check here is a guard, not a service, and
it throws rather than silently computing a shifted answer.
"""
function _build_packed_range(fs::FlipStructure{T}, ::Val{bs};
                             W::Int = PACKED_PARAMS.W, G::Int = PACKED_PARAMS.G) where {T,bs}
    W = W == 0 ? _packed_lane_width(T) : W
    (W > 0 && ispow2(W)) || throw(ArgumentError(
        "spin_flip/packed: W must be a positive power of two (0 = derive), got $W."))
    G >= 1 || throw(ArgumentError("spin_flip/packed: G must be >= 1, got $G."))
    ispow2(bs) || throw(ArgumentError(
        "spin_flip/packed: bs must be a power of two, got $bs. Use spin_flip/rows."))
    bs <= W || throw(ArgumentError(
        "spin_flip/packed: bs = $bs exceeds the pack width W = $W. Use spin_flip/rows."))
    P = W ÷ bs
    fs.dim >= P || throw(ArgumentError(
        "spin_flip/packed: dim = $(fs.dim) is smaller than one pack (P = $P)."))
    while G > 1 && fs.dim < P * G
        G >>= 1
    end
    align = P * G
    masks = Val(Tuple(fs.masks))
    values = SVector{length(fs.values),T}(fs.values)
    diags = fs.diags
    Wv, Gv = Val(W), Val(G)
    return (Y::Matrix{T}, X::Matrix{T}, lo::Int, hi::Int) -> begin
        (lo - 1) % align == 0 || throw(ArgumentError(
            "spin_flip/packed range: lo-1 = $(lo - 1) is not a multiple of P·G = $align. " *
            "Chunk on `range_alignment` boundaries."))
        GC.@preserve Y X diags begin
            _packed_range!(pointer(Y), pointer(X), pointer(diags), values,
                           lo - 1, hi, masks, Wv, Val(bs), Gv)
        end
        return nothing
    end
end
