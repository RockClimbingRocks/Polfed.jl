# The mapping registry: names → row-kernel builders, plus the human-readable listing.
# Adding a kernel later is one entry here + its builder branch (like the filter seam). Stage 1
# ships the three reads kernels; :mul is a benchmark-only reference and :shuf is a placeholder
# for the model-declared flip mapping (built in a later stage).

"""
    RowMappingInfo

One registry row: `id`, `family` (`:reads`/`:flip`/`:ref`), a short `applies` description of the
structure it needs, and the `params` it accepts (`G`, `W`). Used by [`list_mappings`](@ref).
"""
struct RowMappingInfo
    id::Symbol
    family::Symbol
    applies::String
    params::String
end

const REGISTRY = (
    RowMappingInfo(:mul,        :ref,   "any matrix (mul!(Y,H,X) baseline)",   "—"),
    RowMappingInfo(:csr_simd,   :reads, "any structured H (single value)",     "—"),
    RowMappingInfo(:ell32_svec, :reads, "regular row length, L ≤ 30",          "G"),
    RowMappingInfo(:csr64_svec, :reads, "any structured H (single value)",     "G"),
    RowMappingInfo(:shuf,       :flip,  "off-diag positions = σˣ 1-bit flips",  "W, G"),
)

"""
    READS_KERNELS

The reads kernel ids benchmarked/built in Stage 1 (`:mul` is a reference, `:shuf` is a later
flip stage).
"""
const READS_KERNELS = (:csr_simd, :ell32_svec, :csr64_svec)

"""
    build_row_mapping(id::Symbol, s::MappingStructure, ::Val{bs}; G=4, W=8) -> Function

Build the row-layout matvec closure `(Yr, Xr) -> nothing` for kernel `id` over the structure `s`,
specialized to block size `bs` (as `Val`). `Yr`, `Xr` are `bs × D`; the closure computes
`Yr = (H · Xrᵀ)ᵀ`. `G` (ILP width) and `W` (SIMD lanes, flip kernels only) are captured as
compile-time `Val`s so the returned closure is type-stable in the hot loop.

# Throws
- `ArgumentError`: for an unknown or not-yet-implemented kernel id.
"""
function build_row_mapping(id::Symbol, s::MappingStructure, ::Val{bs}; G::Int = 4, W::Int = 8) where {bs}
    if id === :ell32_svec
        Gv = Val(G)
        return (Y, X) -> _row_ell!(Y, X, s.diags, s.hx, s.L, s.nb32, Val(bs), Gv)
    elseif id === :csr64_svec
        Gv = Val(G)
        return (Y, X) -> _row_csr!(Y, X, s.diags, s.hx, s.L, s.flat, s.starts, Val(bs), Gv)
    elseif id === :csr_simd
        return (Y, X) -> _row_csr_simd!(Y, X, s.diags, s.hx, s.L, s.flat, s.starts, Val(bs))
    elseif id === :shuf
        # σˣ flip kernel — assumes the off-diagonals are the L single-bit flips (neighbours
        # computed via xor, NOT read from `s`); only correct for that structure.
        Wv = Val(W); Gv = Val(G)
        return (Y, X) -> _row_shuf!(Y, X, s.diags, s.hx, s.L, Wv, Val(bs), Gv)
    else
        throw(ArgumentError(
            "build_row_mapping: unknown or not-yet-implemented kernel :$id " *
            "(reads: $(join(READS_KERNELS, ", ")); flip: :shuf)."))
    end
end

"""
    list_mappings([io=stdout])

Print the registered row mappings, their family, the structure they apply to, and their tuning
parameters. Value-grouped optimization is worthwhile mainly when the off-diagonals take FEW
distinct values (ideally one).
"""
function list_mappings(io::IO = stdout)
    println(io, "registered row mappings")
    println(io, "(value-grouped optimization pays off when the off-diagonals take FEW distinct")
    println(io, " values — ideally ONE; otherwise prefer the generic path, optimize_mapping=false)")
    println(io)
    @printf(io, " %-12s %-7s %-36s %s\n", "id", "family", "applies to", "params")
    println(io, " ", "─"^70)
    for m in REGISTRY
        @printf(io, " :%-11s %-7s %-36s %s\n", m.id, m.family, m.applies, m.params)
    end
    println(io)
    println(io, " each model declares ONE flip mapping (its champion); σˣ → :shuf.")
    return nothing
end
