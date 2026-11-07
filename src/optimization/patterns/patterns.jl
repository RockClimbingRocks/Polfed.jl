# Off-diagonal patterns. One folder per pattern, each self-contained: its kernels, how to build
# them, and whether they may be used at all.
#
# Adding a pattern is adding a folder here plus one include line — nothing outside patterns/ needs
# to know it exists, because everything reaches it through `can_apply` and `build_kernel`, both
# dispatched on `Val` of the pattern name.

include("verify_kernel.jl")     # shared by every pattern: does the built closure match mul!?

include("spin_flip/spin_flip.jl")   # neighbour computed by xor — no index table
include("hopping/hopping.jl")       # neighbour computed by lattice stride — no index table
include("universal/universal.jl")   # value-grouped CSR — the always-applicable fallback

"""
    can_apply(::Val{pattern}, ::Val{variant}, s::MappingStructure) -> Bool
    why_not(::Val{pattern}, ::Val{variant}, s::MappingStructure) -> String

Per-VARIANT applicability, falling back to the pattern-wide answer.

Most kernels in a pattern accept whatever the pattern accepts, so these defaults forward to the
two-argument methods and a pattern needs no extra code. A kernel with its own restriction — such
as `universal/row_bucketed`, which needs a single value group — adds a three-argument method
beside its own file.
"""
can_apply(p::Val, ::Val, s::MappingStructure) = can_apply(p, s)
why_not(p::Val, ::Val, s::MappingStructure) = why_not(p, s)

"""
    auto_score(::Val{pattern}, ::Val{variant}, bs::Int) -> Int

How much an applicable kernel WANTS to be auto-selected for block size `bs`, when the caller said
`optimize_mapping = true` without naming one. Higher wins; negative means "never pick me
automatically, only if asked for by name".

This exists because applicability and desirability are different questions: `row_bucketed` can run
on any single-group structure, but only beats `row_read` at `bs = 1`, so it scores itself out
everywhere else rather than being excluded by `can_apply`.
"""
auto_score(::Val, ::Val, ::Int) = 0

# ── the range seam ────────────────────────────────────────────────────────────────────────────
# `build_kernel` returns a whole-block closure that owns its own threads. That is exactly what a
# caller doing its own decomposition CANNOT use: `Threads.@threads :static` does not nest. So each
# kernel also exposes a SERIAL range form, and the threading policy moves out to the caller.
#
# The range axis is the kernel's own parallelizable WORK UNIT, not the basis state. For the row
# kernels a unit is one row; for `spin_flip/packed` it is one basis state (in packs of `P·G`); for
# the hopping kernels it is a slice of one lattice line. What every kernel guarantees is that
# distinct unit ranges write DISJOINT parts of `Y`, which is all a decomposition needs.

"""
    supports_range(::Val{pattern}, ::Val{variant}) -> Bool

Whether this kernel exposes a serial range form. **Opt-in**: the default is `false`, so a kernel
that has not been split stays safe rather than silently gaining a wrong one.

`universal/row_bucketed` is the one registered kernel that says `false`. It opens one parallel
region per length bucket plus a tail, so its work does not form a single flat range without
restructuring — and it is only ever auto-selected at `bs = 1`, where there is nothing to split.
"""
supports_range(::Val, ::Val) = false

"""
    build_range_kernel(::Val{pattern}, ::Val{variant}, s::MappingStructure, ::Val{bs}; params...)

The SERIAL range closure `(Y, X, lo, hi) -> nothing` over work units `lo:hi` (1-based, inclusive).
Opens no parallel region — the caller owns the threads.

Pair it with [`range_units`](@ref) for the unit count and [`range_alignment`](@ref) for the
multiple that chunk boundaries must respect.
"""
build_range_kernel(p::Val, v::Val, ::MappingStructure, ::Val; kwargs...) = throw(ArgumentError(
    "build_range_kernel: $(_valname(p))/$(_valname(v)) has no serial range form " *
    "(supports_range is false). Use build_kernel, which owns its own threading."))

"""
    range_units(::Val{pattern}, ::Val{variant}, s::MappingStructure, ::Val{bs}; params...) -> Int

Total number of work units, i.e. the largest valid `hi` for [`build_range_kernel`](@ref).
"""
range_units(p::Val, v::Val, ::MappingStructure, ::Val; kwargs...) = throw(ArgumentError(
    "range_units: $(_valname(p))/$(_valname(v)) has no serial range form."))

"""
    range_alignment(::Val{pattern}, ::Val{variant}, s::MappingStructure, ::Val{bs}; params...) -> Int

The multiple that every chunk boundary must fall on: a caller must split `1:range_units(...)` so
each chunk's `lo - 1` is divisible by this. `1` means any split is valid.

Only `spin_flip/packed` returns more than `1` — it steps in whole packs, and its lane permutation
is a permutation only on a pack boundary. That is a correctness precondition, not a preference,
and the range closure throws rather than quietly computing a shifted answer.
"""
range_alignment(::Val, ::Val, ::MappingStructure, ::Val; kwargs...) = 1

"""    _valname(::Val{N}) -> N — the symbol inside a `Val`, for error messages."""
_valname(::Val{N}) where {N} = N
