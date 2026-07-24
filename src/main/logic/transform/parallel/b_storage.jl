# Workspace allocation for the parallel Clenshaw apply. The recurrence needs a triplet of
# scratch blocks (`b1, b2, b3`); how many and of what shape depends on the parallelization
# strategy chosen by the mapping stage (see mapping/parallel.jl). This file allocates that
# workspace; clenshaw_apply.jl consumes it. The strategy *types* are defined in the mapping
# slice — here we only dispatch on them.

"""
    get_b_storage(strategy::Parallelization, pu::ProcessingUnit, x0::AbstractVecOrMat)

Allocate the Clenshaw recurrence workspace matching `strategy`:

- [`NoParallel`](@ref) / [`MulRowsParallel`](@ref): one whole-block triplet
  (`3 × similar(x0)`).
- [`MulColsParallel`](@ref): a whole-block triplet for a vector `x0` or when
  `nt_per_col > 1`; otherwise per-column vector triplets (for the threaded path).
- [`TwoLevelParallel`](@ref): per-column vector triplets allocated through `pu`.

The shape must match the [`clenshaw_apply`](@ref) method the same `strategy` selects.
"""
get_b_storage(::NoParallel, ::ProcessingUnit, x0::AbstractVecOrMat) = [similar(x0) for _ in 1:3]
get_b_storage(::MulRowsParallel, ::ProcessingUnit, x0::AbstractVecOrMat) = [similar(x0) for _ in 1:3]

get_b_storage(::MulColsParallel, ::ProcessingUnit, x0::AbstractVector) = [similar(x0) for _ in 1:3]

function get_b_storage(strategy::MulColsParallel, pu::ProcessingUnit, x0::AbstractMatrix)
    strategy.nt_per_col > 1 && return [similar(x0) for _ in 1:3]
    return _per_column_triplets(pu, x0)
end

get_b_storage(::TwoLevelParallel, pu::ProcessingUnit, x0::AbstractVecOrMat) = _per_column_triplets(pu, x0)

"""
    _per_column_triplets(pu::ProcessingUnit, x0::AbstractVecOrMat)

A length-`ncols` vector of 3-vector recurrence workspaces, one triplet per block column,
allocated with `pu`'s vector type (so GPU runs get device vectors). Used by the column-
threaded [`MulColsParallel`](@ref) and the distributed [`TwoLevelParallel`](@ref) paths.
"""
function _per_column_triplets(pu::ProcessingUnit, x0::AbstractVecOrMat)
    E = eltype(x0)
    dim = size(x0, 1)
    ncols = size(x0, 2)
    return [[pu.Vector{E}(undef, dim) for _ in 1:3] for _ in 1:ncols]
end

"""
    get_clenshaw_shared_buffer!(ps::TwoLevelParallel, ::Type{T}, dims) -> SharedArray{T,2}

Return a reusable shared-memory output buffer for the two-level-parallel Clenshaw apply,
(re)allocating it on `ps` when the element type or `dims` change. The buffer is where the
Distributed workers write their per-column results before they are copied back into `Y`
(see the [`TwoLevelParallel`](@ref) method of [`clenshaw_apply`](@ref)).
"""
@inline function get_clenshaw_shared_buffer!(ps::TwoLevelParallel, ::Type{T}, dims::Tuple{Int,Int}) where {T<:Number}
    buffer = ps.clenshaw_shared_buffer
    if !(buffer isa SharedArray{T,2}) || size(buffer) != dims
        ps.clenshaw_shared_buffer = SharedArray{T}(dims)
    end
    return ps.clenshaw_shared_buffer::SharedArray{T,2}
end
