# Machinery both hopping kernels share: the dimension-specialisation boundary, the per-line
# neighbour bases, and the work decomposition.
#
# Both kernels have the SAME loop skeleton and differ only in what a "register" holds:
#
#   packed  — P = W/bs adjacent SITES at fixed block index. Axis-1 neighbours straddle a register
#             boundary, so they cost two `shufflevector`s of (prev,cur) and (cur,next).
#   rows    — one site's bs blocks in an `SVector{bs}`. An axis-1 neighbour is then a WHOLE
#             register, so sliding is a plain rotation and there is no shuffle at all.
#
# Either way axis 1 is slid: `cur` becomes the next step's `prev` and the loaded `next` becomes
# its `cur`, so each site is loaded ONCE for all three of its axis-1 roles. That takes the
# 2D+1-point stencil from 2D+1 input streams down to 2D-1, which is the whole reason this
# pattern exists rather than a generic row kernel over a neighbour table.
#
# The other 2(D-1) neighbours are ordinary strided loads whose LINE bases are hoisted out of the
# x loop — computed once per line, not once per site.

"""
    MAX_UNROLLED_DIM

Up to this lattice dimension the per-site neighbour sum is fully UNROLLED at specialization
time — every one of the `2(D-1)` orthogonal loads is emitted as straight-line code with an
immediate offset, so the hot loop contains no loop over axes, no coordinate reconstruction, and
no indirection. Above it the same loads run as a counted loop over the (still statically sized)
tuple of line bases.

The cut exists to bound CODE, not to bound correctness: `D = 12` would emit 22 live load streams
plus their accumulator tree into one loop body and spill. Everything above the cut still carries
`D` as a type parameter and still hoists all coordinate work per line — only the innermost sum
becomes a loop.

Measured basis for placing it at 6, on Zen 4 (`notebooks/benchmarks/matvec/hop`): a generated
per-`D` kernel tied hand-written fixed-`D` kernels at a median ratio of 1.000 over `D = 1…6`,
with outliers in BOTH directions (`D=6, bs=4` was 0.89x, i.e. the generated one was faster).
Hand-writing one kernel per dimension buys nothing measurable; specialising the compiler's view
of `D` buys 3.7-5.7x.
"""
const MAX_UNROLLED_DIM = 6

"""    _hop_unrolled(N::Int) -> Bool — is a stencil with `N` orthogonal neighbours unrolled?"""
@inline _hop_unrolled(N::Int) = N <= 2 * (MAX_UNROLLED_DIM - 1)

"""    _hop_byteoffset(::Type{T}, n) -> Int — element count to byte count, for raw pointers."""
@inline _hop_byteoffset(::Type{T}, n::Int) where {T} = n * sizeof(T)

"""
    _hop_pairwise_add(terms) -> Expr

Sum generated terms as a balanced tree rather than a left-leaning chain, so the `2(D-1)`
neighbour loads accumulate through `log₂` dependent adds instead of `2(D-1)` of them. At `D = 6`
that is depth 4 instead of 10 on the critical path of every site.
"""
function _hop_pairwise_add(terms::Vector)
    isempty(terms) && return nothing
    level = terms
    while length(level) > 1
        nxt = Any[]
        i = 1
        while i <= length(level)
            if i == length(level)
                push!(nxt, level[i])
            else
                push!(nxt, :($(level[i]) + $(level[i + 1])))
            end
            i += 2
        end
        level = nxt
    end
    return level[1]
end

"""
    _hop_line_bases(shape, strides, line) -> NTuple{2(D-1),Int}

The `2(D-1)` orthogonal neighbour LINE bases (0-based site indices) of x-line `line`, in the
order `(axis2⁻, axis2⁺, axis3⁻, axis3⁺, …)`. Empty for `D = 1`.

An x-line is a maximal run of sites differing only in the axis-1 coordinate, so `line` indexes
the lattice with axis 1 collapsed: `line = Σ_{k≥2} x_k · (s_k / L₁)`. Every axis-1 site in the
line shares these bases, which is why the periodic wrap tests — one per axis, per line — never
appear in the x loop.

Generated over `D` for every dimension, unrolled cut or not: this runs once per line, so
unrolling it is always right.
"""
@generated function _hop_line_bases(shape::NTuple{D,Int}, strides::NTuple{D,Int},
                                    line::Int) where {D}
    D == 1 && return :(())
    setup = Expr[:(base = line * shape[1])]
    names = Symbol[]
    for k in 2:D
        sk, Lk, ck = Symbol(:s_, k), Symbol(:L_, k), Symbol(:c_, k)
        mk, pk = Symbol(:m_, k), Symbol(:p_, k)
        push!(setup, :($sk = strides[$k]))
        push!(setup, :($Lk = shape[$k]))
        # strides[k] / L₁ = prod(L₂ … L_{k-1}) — the line-index divisor for this axis
        push!(setup, :($ck = rem(div(line, div($sk, shape[1])), $Lk)))
        push!(setup, :($mk = $ck == 0 ? base + ($Lk - 1) * $sk : base - $sk))
        push!(setup, :($pk = $ck == $Lk - 1 ? base - ($Lk - 1) * $sk : base + $sk))
        push!(names, mk, pk)
    end
    return quote
        @inbounds begin
            $(setup...)
            $(Expr(:tuple, names...))
        end
    end
end

# ---------------------------------------------------------------------------------------------
# Work decomposition.
#
# The unit of work is (one x-line, a sub-range of its cells). Splitting INSIDE a line is not an
# optimisation: at D = 1 the lattice is a single line, so line-only chunking would run the whole
# matvec on one thread. A split range re-primes its own `prev` from the cell before it and takes
# the periodic wrap only if it owns the end of the line, so the slide is unaffected.
# ---------------------------------------------------------------------------------------------

"""
    _hop_xsplit(nlines, ncells, nt) -> Int

How many pieces each x-line is cut into. `1` whenever there are already comfortably more lines
than threads (the common case, `D ≥ 2`); otherwise enough pieces to give every thread work, which
is what keeps `D = 1` and small lattices parallel at all.
"""
@inline function _hop_xsplit(nlines::Int, ncells::Int, nt::Int)
    (nt <= 1 || nlines >= 4nt) && return 1
    return clamp(cld(4nt, nlines), 1, ncells)
end

"""
    _hop_unit_range(u, xsplit, ncells) -> (line, lo, hi)

Decode work unit `u` (0-based) into its line and its half-open cell range `lo:hi-1`.
"""
@inline function _hop_unit_range(u::Int, xsplit::Int, ncells::Int)
    line, k = divrem(u, xsplit)
    return line, div(k * ncells, xsplit), div((k + 1) * ncells, xsplit)
end

"""
    _hop_chunking(nunits, xsplit, tile, nt) -> (chunk, nchunks)

How the units are handed to threads. `tile = 0` gives each thread ONE contiguous range (the
default: fewest boundaries, best locality for the orthogonal streams). `tile > 0` pins the chunk
to that many x-lines instead, which trades locality for load balance — worth sweeping when the
lattice is anisotropic or the thread count does not divide the line count.
"""
@inline function _hop_chunking(nunits::Int, xsplit::Int, tile::Int, nt::Int)
    chunk = tile == 0 ? cld(nunits, max(nt, 1)) : tile * xsplit
    chunk = clamp(chunk, 1, max(nunits, 1))
    return chunk, cld(nunits, chunk)
end
