# The three general "reads" kernels (row layout, neighbour indices READ from memory).
# All compute, per block column b and row i:  Y[b,i] = diags[i]*X[b,i] + hx * Σ_c X[b, nb(i,c)].
# They differ on the two axes that matter per architecture:
#
#   csr_simd    : CSR (flat+starts, Int64), plain @simd over the contiguous block — NO StaticArrays.
#   ell32_svec  : ELL (Int32 indices),      SVector-`G` register accumulation + G-way ILP.
#   csr64_svec  : CSR (flat+starts, Int64),  SVector-`G` register accumulation + G-way ILP.
#
# `csr_simd` vs `csr64_svec` isolates accumulation strategy (same storage); `ell32_svec` vs
# `csr64_svec` isolates storage/index width (same accumulation). Ported from test-10-fable.
# Threading: `Threads.@threads :static` over `Threads.nthreads()` (fixed per process).

# ── ell32_svec: Int32 ELL + SVector-G ─────────────────────────────────────────
# nb[(i-1)*L + c] = c-th (1-based) neighbour column of row i.
@generated function _ell_blockG!(Y, X, diags::AbstractVector{T}, hx, i0::Int, L::Int,
                                 nb::AbstractVector{TI}, ::Val{bs}, ::Val{G}) where {T,TI<:Integer,bs,G}
    init = Expr[]; updates = Expr[]; stores = Expr[]
    for g in 1:G
        a = Symbol(:a, g); base = Symbol(:base, g); off = g - 1
        push!(init, :($a = @inbounds(diags[i0 + $off]) * loadv(X, i0 + $off, Val(bs), T)))
        push!(init, :($base = (i0 + $off - 1) * L))
        push!(updates, :($a = muladd(hx, loadv(X, Int(@inbounds(nb[$base + c])), Val(bs), T), $a)))
        push!(stores, :(storev!(Y, i0 + $off, $a)))
    end
    quote
        $(init...)
        @inbounds for c in 1:L
            $(updates...)
        end
        $(stores...)
        return nothing
    end
end

function _row_ell!(Y, X, diags::AbstractVector{T}, hx, L, nb, ::Val{bs}, ::Val{G}) where {T,bs,G}
    D = size(X, 2)
    length(nb) == D * L || error("ELL structure size mismatch")
    hxT = T(hx)
    nt, chunk = _chunks_aligned(D, G)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, D)
        i = lo
        while i + G - 1 <= hi
            _ell_blockG!(Y, X, diags, hxT, i, L, nb, Val(bs), Val(G))
            i += G
        end
        while i <= hi
            _ell_blockG!(Y, X, diags, hxT, i, L, nb, Val(bs), Val(1))
            i += 1
        end
    end
    return nothing
end

# ── csr64_svec: Int64 CSR + SVector-G ─────────────────────────────────────────
@generated function _csr_blockG!(Y, X, diags::AbstractVector{T}, hx, i0::Int, L::Int,
                                 flat::AbstractVector{TI}, starts::AbstractVector{Int},
                                 ::Val{bs}, ::Val{G}) where {T,TI<:Integer,bs,G}
    init = Expr[]; updates = Expr[]; stores = Expr[]
    for g in 1:G
        a = Symbol(:a, g); s = Symbol(:s, g); off = g - 1
        push!(init, :($a = @inbounds(diags[i0 + $off]) * loadv(X, i0 + $off, Val(bs), T)))
        push!(init, :($s = @inbounds(starts[i0 + $off])))
        push!(updates, :($a = muladd(hx, loadv(X, Int(@inbounds(flat[$s + c])), Val(bs), T), $a)))
        push!(stores, :(storev!(Y, i0 + $off, $a)))
    end
    quote
        $(init...)
        @inbounds for c in 0:L-1
            $(updates...)
        end
        $(stores...)
        return nothing
    end
end

function _row_csr!(Y, X, diags::AbstractVector{T}, hx, L, flat, starts, ::Val{bs}, ::Val{G}) where {T,bs,G}
    D = size(X, 2)
    hxT = T(hx)
    nt, chunk = _chunks_aligned(D, G)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, D)
        i = lo
        while i + G - 1 <= hi
            _csr_blockG!(Y, X, diags, hxT, i, L, flat, starts, Val(bs), Val(G))
            i += G
        end
        while i <= hi
            _csr_blockG!(Y, X, diags, hxT, i, L, flat, starts, Val(bs), Val(1))
            i += 1
        end
    end
    return nothing
end

# ── csr_simd: CSR + plain @simd over the block (no StaticArrays) ───────────────
function _row_csr_simd!(Y, X, diags::AbstractVector{T}, hx, L, flat, starts, ::Val{bs}) where {T,bs}
    D = size(X, 2)
    hxT = T(hx)
    nt, chunk = _chunks_aligned(D, 1)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, D)
        @inbounds for i in lo:hi
            di = diags[i]
            @simd for b in 1:bs
                Y[b, i] = di * X[b, i]
            end
            s = starts[i]
            for c in 0:L-1
                j = Int(flat[s + c])
                @simd for b in 1:bs
                    Y[b, i] += hxT * X[b, j]
                end
            end
        end
    end
    return nothing
end
