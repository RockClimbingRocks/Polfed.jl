# Flip kernels: neighbours computed on the fly via xor (no index reads). Specialized per
# neighbour pattern and declared by the model — Stage-1 ships `shuf` for the σˣ single-site
# pattern (neighbours = all single-bit flips `xor(site, 1<<bit)`), the benchmark champion.
#
# `shuf` packs P = W÷bs consecutive Hilbert columns into one `Vec{W,T}` SIMD register: if the
# pack base is P-aligned, those P columns differ only in the lowest log₂P bits of the site
# index, so low-bit flips are in-register `shufflevector`s (zero loads) and high-bit flips are
# one aligned vector load per bit. G packs use independent accumulators (ILP). Float64/Float32
# only (the W lanes are reals). Ported verbatim from the test-10-fable benchmark.
#
# ASSUMES the σˣ 1-bit-flip structure — correct only when H's off-diagonals are exactly the L
# single-bit flips with value `hx` (e.g. QREM/QSun). The registry gates it accordingly.

# One inner block: G packs of P = W÷bs columns each, from 0-based site p0 (a multiple of P).
@generated function _shuf_packG!(pY::Ptr{T}, pX::Ptr{T}, pd::Ptr{T}, hx::T, p0::Int, L::Int,
                                 ::Val{W}, ::Val{bs}, ::Val{G}) where {T,W,bs,G}
    P = W ÷ bs
    nlow = trailing_zeros(P)          # bits handled by in-register shuffles
    sz = sizeof(T)
    init = Expr[]; shufs = Expr[]; upds = Expr[]; stores = Expr[]
    for g in 0:G-1
        v = Symbol(:v, g); a = Symbol(:a, g); dv = Symbol(:d, g); eoff = g * W
        push!(init, :($v = vload(Vec{$W,T}, pX + (p0 * $bs + $eoff) * $sz)))
        if bs == 1
            push!(init, :($dv = vload(Vec{$W,T}, pd + (p0 + $(g * P)) * $sz)))
        else
            # lane w belongs to column p0 + g*P + (w-1)÷bs (0-based site)
            push!(init, :($dv = Vec{$W,T}($(Expr(:tuple,
                [:(unsafe_load(pd, p0 + $(g * P + (w - 1) ÷ bs) + 1)) for w in 1:W]...)))))
        end
        push!(init, :($a = $dv * $v))
        for t in 0:nlow-1
            mask = ntuple(w -> xor(w - 1, bs << t), W)      # bit t swaps lane blocks of size bs·2^t
            push!(shufs, :($a = muladd(hv, shufflevector($v, Val($mask)), $a)))
        end
        push!(upds, :($a = muladd(hv,
            vload(Vec{$W,T}, pX + (xor(p0 + $(g * P), m) * $bs) * $sz), $a)))
        push!(stores, :(vstore($a, pY + (p0 * $bs + $eoff) * $sz)))
    end
    quote
        hv = Vec{$W,T}(hx)
        $(init...)
        $(shufs...)
        for bit in $nlow:L-1
            m = 1 << bit
            $(upds...)
        end
        $(stores...)
        return nothing
    end
end

"""
    _row_shuf!(Y, X, diags, hx, L, ::Val{W}, ::Val{bs}, ::Val{G}) -> nothing

Row-layout σˣ matvec via column-packed SIMD shuffles. `Y`, `X` are `bs × D` `Matrix`es; `diags`
is length `D`. Requires `bs`, `W` powers of two with `bs ≤ W` (so `P = W÷bs ≥ 1`), `D` a multiple
of `P`, and `L ≥ log₂P`.
"""
function _row_shuf!(Y::Matrix{T}, X::Matrix{T}, diags::Vector{T}, hx, L,
                    ::Val{W}, ::Val{bs}, ::Val{G}) where {T,W,bs,G}
    D = size(X, 2)
    P = W ÷ bs
    P ≥ 1 && P * bs == W && ispow2(bs) || error("bs=$bs incompatible with W=$W")
    D % P == 0 || error("D=$D not a multiple of P=$P")
    L ≥ trailing_zeros(P) || error("L=$L too small for P=$P")
    size(Y) == size(X) && length(diags) == D || error("shape mismatch")
    hxT = T(hx)
    nt, chunk = _chunks_aligned(D, P * G)
    GC.@preserve Y X diags begin
        pY = pointer(Y); pX = pointer(X); pd = pointer(diags)
        Threads.@threads :static for t in 1:nt
            p0 = (t - 1) * chunk               # 0-based site, multiple of P·G
            hi = min(t * chunk, D)
            while p0 + G * P <= hi
                _shuf_packG!(pY, pX, pd, hxT, p0, L, Val(W), Val(bs), Val(G))
                p0 += G * P
            end
            while p0 + P <= hi                 # tail: one pack at a time
                _shuf_packG!(pY, pX, pd, hxT, p0, L, Val(W), Val(bs), Val(1))
                p0 += P
            end
        end
    end
    return nothing
end
