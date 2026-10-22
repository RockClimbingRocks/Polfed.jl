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
- [`BlockWise`](@ref): a single whole-block call (the decomposition, when present, lives
  inside the expansion's matvec). Any evaluator `expansion(Y, X, b)` — including the fused
  [`AxpbyClenshaw`](@ref) / [`RangeClenshaw`](@ref) — is accepted here.
- [`VectorWise`](@ref) on [`CPU`](@ref): threaded over columns with per-column workspace,
  or — for a vector input, which is a single column — one BLAS-serialized call.
"""
function clenshaw_apply(expansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVecOrMat}, ::ProcessingUnit, ::BlockWise,
                        ::Layout = ColumnLayout())
    expansion(Y, X, b)
    return nothing
end

"""CPU vector overload for [`VectorWise`](@ref): a vector input is a single column, so there
is nothing to split — one call, with BLAS serialized so the expansion's optimized kernels own
the threads. Selected by the workspace shape (three arrays, not per-column triplets)."""
function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVecOrMat}, ::CPU, ::VectorWise,
                        ::Layout = ColumnLayout())
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        expansion(Y, X, b)
    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
    return nothing
end

"""CPU threaded per-vector overload for [`VectorWise`](@ref): one Krylov vector per thread,
each with its own recurrence triplet. The vector axis is selected by `layout` through
[`_vector_slice`](@ref) — columns under [`ColumnLayout`](@ref), rows under
[`RowLayout`](@ref) — so a row-layout block is split into `bs` vectors of length `D`, not
into `D` fragments of length `bs`.

Selected by the workspace being a vector of TRIPLETS (`b[i]` is itself a 3-element container),
where the whole-block method takes a bare 3-element triplet. The inner buffers are
`AbstractVecOrMat`, not `AbstractVector`: under [`RowLayout`](@ref) a single Krylov vector's
buffer is a `1 × D` matrix, and requiring `AbstractVector` there sent row-layout runs to the
whole-block method instead, which then failed with an unreadable `MethodError`."""
function clenshaw_apply(expansion::ClenshawExpansion, Y::AbstractVecOrMat, X::AbstractVecOrMat,
                        b::AbstractVector{<:AbstractVector{<:AbstractVecOrMat}}, ::CPU, ::VectorWise,
                        layout::Layout = ColumnLayout())
    @assert size(X) == size(Y) "Y and X must have the same size"
    nvecs = blocksize(layout, X)
    @assert length(b) == nvecs "workspace length must equal the number of Krylov vectors"

    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        Threads.@threads for i in 1:nvecs
            expansion(_vector_slice(layout, Y, i), _vector_slice(layout, X, i), b[i])
        end
    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
    return nothing
end
