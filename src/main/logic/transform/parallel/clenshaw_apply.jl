# Apply a built ClenshawExpansion to a block under a chosen Parallelization. This is the
# bridge between the transform (which builds the expansion) and the parallel strategy types
# (which say how the block is split). The strategy-dispatched matvec itself lives inside the
# expansion's `f!_rescaled`; here we only choose the column/worker decomposition and gate
# BLAS threads. Workspace comes from get_b_storage (b_storage.jl), shaped to match.

"""
    clenshaw_apply(expansion, Y, X, b, pu::ProcessingUnit, strategy::Parallelization) -> nothing

Apply the Clenshaw `expansion` to `X`, writing `Y`, under the parallelization `strategy`.
`b` is the recurrence workspace from [`get_b_storage`](@ref) (shape matched to `strategy`).

Methods:
- [`NoParallel`](@ref) / [`MulRowsParallel`](@ref): a single whole-block call (row-level
  parallelism, when present, lives inside the expansion's matvec).
- [`MulColsParallel`](@ref) on [`CPU`](@ref): threaded over columns (per-column workspace),
  or a single BLAS-serialized whole-block call when `nt_per_col > 1`.
- [`TwoLevelParallel`](@ref) on [`CPU`](@ref): columns distributed over the worker pool into
  a shared output buffer.
"""
function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVecOrMat}, ::ProcessingUnit, ::NoParallel)
    expansion(Y, X, b)
    return nothing
end

function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVecOrMat}, ::ProcessingUnit, ::MulRowsParallel)
    expansion(Y, X, b)
    return nothing
end

"""CPU whole-block overload for [`MulColsParallel`](@ref) with `nt_per_col > 1`: serialize
BLAS so the expansion's optimized kernels own the threads."""
function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVecOrMat}, ::CPU, ::MulColsParallel)
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        expansion(Y, X, b)
    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
    return nothing
end

"""CPU threaded-column overload for [`MulColsParallel`](@ref): one column per thread, each
with its own recurrence triplet."""
function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVector{<:AbstractVector}}, ::CPU, ::MulColsParallel)
    @assert size(X) == size(Y) "Y and X must have the same size"
    @assert length(b) == size(X, 2) "workspace length must equal the number of columns"

    ncols = size(X, 2)
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in 1:ncols
            expansion(view(Y, :, i), view(X, :, i), b[i])
        end
    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
    return nothing
end

"""CPU distributed-worker overload for [`TwoLevelParallel`](@ref): columns are mapped over
the worker pool into a shared output buffer, then copied back into `Y`."""
function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVector{<:AbstractVector}}, ::CPU, ps::TwoLevelParallel)
    nvecs = size(X, 2)
    @assert size(X) == size(Y) "Y and X must have the same size"
    @assert length(b) == nvecs "workspace length must equal the number of columns"

    Y_sh = get_clenshaw_shared_buffer!(ps, eltype(Y), size(Y))
    pmap(i -> expansion(view(Y_sh, :, i), view(X, :, i), b[i]), ps.worker_pool, 1:nvecs)
    Y .= Y_sh
    return nothing
end
