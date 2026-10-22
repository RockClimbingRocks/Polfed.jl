"""Upper bound on the entry slots reserved by `sizehint!` (pointers only, ~8 B each)."""
const _VECTORBASIS_SIZEHINT_CAP = 1024

"""
    VectorBasis{T,E}(maxdim::Int, x0::AbstractVecOrMat, pu::ProcessingUnit)

Vector-of-arrays basis storage backend for Lanczos vectors/blocks, grown **on demand**.

Unlike [`MatrixBasis`](@ref), which preallocates the full `dim × maxdim` (or `maxdim × dim`)
block up front, this backend starts empty and allocates one entry per [`add!`](@ref). Storage
is therefore proportional to the iterations actually performed, not to the budget, so `maxdim`
can be set generously without paying for it: a run that converges in 150 iterations holds 150
vectors even if the cap was 5000.

`maxdim` is consequently a **capacity hint only** — it sizes the entry-pointer array via
`sizehint!` (bounded, so an enormous cap costs nothing) and never limits how much is stored.
The iteration cap still comes from the convergence criterion, which is what stops the run.

# Notes
- `nvecs` counts whole **entries** (blocks of `blocksize` vectors), whereas `MatrixBasis.nvecs`
  counts individual vectors. [`FactorizationReport`](@ref) normalizes the two.
- With growth there is no capacity to overrun, so — unlike the fixed-size backend — nothing is
  ever shifted out. That matters for correctness: silently dropping the oldest vector would
  leave full reorthogonalization projecting against an incomplete basis.
- Column semantics only; [`lanczos`](@ref) rejects `VectorBasis` with [`RowLayout`](@ref).
"""
mutable struct VectorBasis{T,E} <: OrthonormalBasis
    basis::Vector{T}
    nvecs::Int
    blocksize::Int

    function VectorBasis{E}(maxdim::Int, x0::T, ::ProcessingUnit,
                            layout::Layout=ColumnLayout()) where {T<:AbstractVecOrMat, E<:Number}
        bs = blocksize(layout, x0)
        basis = Vector{T}()
        # Reserve only the pointer array, and only a bounded amount of it: `maxdim` is allowed to
        # be huge here (that is the point of this backend), and `cld(maxdim, bs)` pointers would
        # otherwise reintroduce a cost that scales with the cap rather than with the work done.
        sizehint!(basis, min(cld(maxdim, bs), _VECTORBASIS_SIZEHINT_CAP))
        new{T,E}(basis, 0, bs)
    end
end

"""
    add!(basis::VectorBasis, v) -> nothing

Append a vector/block to basis storage, allocating one new entry for it.

The entry is a fresh array copied from `v` (never an alias): the caller reuses its work buffers
between iterations, so storing `v` itself would leave every basis entry pointing at the same
memory. `similar(v)` also keeps the entry on the same device as `v` on the GPU path.
"""
function add!(basis::VectorBasis{T,E}, v::T) where {T,E}
    entry = similar(v)
    entry .= v
    push!(basis.basis, entry)
    basis.nvecs += 1
    return nothing
end

"""
    allvecs(basis::VectorBasis)

Return currently stored basis entries.

Slices to a concrete `Vector` rather than a view: `_reconstruct_eigenvectors!` dispatches on
`Vector{<:AbstractVector}` / `Vector{<:AbstractMatrix}`, which a `SubArray` would not match.
The copy is of the entry pointers only (`nvecs` words), not of the basis vectors.
"""
function allvecs(basis::VectorBasis)
    basis.nvecs == 0 && throw(ArgumentError("Basis is empty"))
    return basis.basis[1:basis.nvecs]
end

"""
    all_withoutlasttwo(basis::VectorBasis)

Return basis entries excluding the most recent two blocks. `nvecs` counts whole
blocks (entries) here, so the two most recent blocks are simply the last two entries.
"""
function all_withoutlasttwo(basis::VectorBasis)
    basis.nvecs < 2 && throw(ArgumentError("Not enough vectors in the basis"))
    return basis.basis[1:basis.nvecs-2]
end
