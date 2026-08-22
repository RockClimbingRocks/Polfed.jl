# Workspace allocation for the parallel Clenshaw apply. The recurrence needs a triplet of
# scratch blocks (`b1, b2, b3`); how many and of what shape depends on the parallelization
# strategy chosen by the mapping stage (see mapping/parallel.jl). This file allocates that
# workspace; clenshaw_apply.jl consumes it. The strategy *types* are defined in the mapping
# slice — here we only dispatch on them.

"""
    get_b_storage(strategy::Parallelization, pu::ProcessingUnit, x0::AbstractVecOrMat,
                  layout::Layout = ColumnLayout())

Allocate the Clenshaw recurrence workspace matching `strategy`:

- [`BlockWise`](@ref): one whole-block triplet (`3 × similar(x0)`) — the split happens
  inside the operator call, so the workspace is shared and the layout is irrelevant here.
- [`VectorWise`](@ref): one vector triplet per Krylov vector, so each thread owns its own
  buffers. A vector `x0` is a single Krylov vector and gets the plain whole-vector triplet.

The shape must match the [`clenshaw_apply`](@ref) method the same `strategy` selects.
"""
get_b_storage(::BlockWise, ::ProcessingUnit, x0::AbstractVecOrMat, ::Layout = ColumnLayout()) =
    [similar(x0) for _ in 1:3]

get_b_storage(::VectorWise, ::ProcessingUnit, x0::AbstractVector, ::Layout = ColumnLayout()) =
    [similar(x0) for _ in 1:3]
get_b_storage(::VectorWise, pu::ProcessingUnit, x0::AbstractMatrix,
              layout::Layout = ColumnLayout()) = _per_vector_triplets(pu, x0, layout)


"""
    _per_vector_triplets(pu::ProcessingUnit, x0::AbstractMatrix, layout::Layout)

A length-`block_size` vector of 3-buffer recurrence workspaces, one triplet per Krylov vector,
allocated through `pu` (so GPU runs get device arrays). Used by the [`VectorWise`](@ref) path,
where each thread owns its buffers outright.

The count and length come from [`blocksize`](@ref)/[`opdim`](@ref), not from a fixed axis: under
[`RowLayout`](@ref) the block is `bs × D`, so reading `size(x0, 1)` as the dimension would
allocate `D` triplets of length `bs` — the problem exactly transposed.

The buffer SHAPE follows the layout for the same reason [`_vector_slice`](@ref) does: the
workspace is handed to the same operator as the slice, so a row-layout run needs `1 × D`
matrices, not length-`D` vectors. A vector where the operator expects a one-row block does not
merely underperform — `mul!` has no method for it.
"""
function _per_vector_triplets(pu::ProcessingUnit, x0::AbstractMatrix, layout::Layout)
    E = eltype(x0)
    dim = opdim(layout, x0)
    nvecs = blocksize(layout, x0)
    return [[_vector_buffer(pu, layout, E, dim) for _ in 1:3] for _ in 1:nvecs]
end

"""
    _vector_buffer(pu, layout, ::Type{E}, dim) -> AbstractVecOrMat{E}

One recurrence buffer for a single Krylov vector, shaped for `layout`'s operator: a length-`dim`
vector under [`ColumnLayout`](@ref), a `1 × dim` matrix under [`RowLayout`](@ref).
"""
_vector_buffer(pu::ProcessingUnit, ::ColumnLayout, ::Type{E}, dim::Int) where {E} =
    pu.Vector{E}(undef, dim)
_vector_buffer(pu::ProcessingUnit, ::RowLayout, ::Type{E}, dim::Int) where {E} =
    pu.Matrix{E}(undef, 1, dim)

