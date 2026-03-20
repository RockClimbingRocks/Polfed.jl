
const ClenshawType = Clenshaw

"""
    get_clenshaw_shared_buffer!(parallel_strategy::TwoLevelParallel, ::Type{T}, dims::Tuple{Int,Int}) where {T<:Number} -> SharedArray{T,2}

Return a reusable shared-memory buffer for two-level-parallel Clenshaw output.

The buffer is allocated (or resized) lazily and stored in
`parallel_strategy.clenshaw_shared_buffer`.
"""
@inline function get_clenshaw_shared_buffer!(
    parallel_strategy::TwoLevelParallel,
    ::Type{T},
    dims::Tuple{Int,Int},
) where {T<:Number}
    buffer = parallel_strategy.clenshaw_shared_buffer
    if !(buffer isa SharedArray{T,2}) || size(buffer) != dims
        parallel_strategy.clenshaw_shared_buffer = SharedArray{T}(dims)
    end
    return parallel_strategy.clenshaw_shared_buffer::SharedArray{T,2}
end




"""
    clenshaw(transform::ClenshawType, Y, X, b, pu, parallel_strategy) -> nothing

Apply a [`Clenshaw`](@ref) transform using the selected processing unit and
parallelization strategy.

Methods:
- `NoParallel`: direct call.
- `MulColsParallel` on CPU: threaded over columns.
- `TwoLevelParallel` on CPU: distributed over worker pool with shared output.
"""
function clenshaw(
    transform::ClenshawType, 
    W::AbstractVecOrMat{T},
    V::AbstractVecOrMat{T}, 
    b::AbstractVector{<:AbstractVecOrMat{T}}, 
    ::ProcessingUnit, 
    ::NoParallel
) where {T<:Number}

    transform(W, V, b)
end




"""CPU threaded-column overload of `clenshaw` for [`MulColsParallel`](@ref)."""
function clenshaw(
    transform::ClenshawType, 
    Y::AbstractVecOrMat{<:Number}, 
    X::AbstractVecOrMat{<:Number}, 
    b::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}}, 
    ::CPU, 
    ::MulColsParallel
)
    @assert size(X) == size(Y) "Y and Ỹ must be of the same size!"
    @assert length(b) == size(X, 2) "Length of b must be the same as the number of columns of Y!"
    
    ncols = size(X, 2)  # Number of columns
    prev_blas_threads = BLAS.get_num_threads(); BLAS.set_num_threads(1)
    @threads for i in 1:ncols
        transform(view(Y, :, i), view(X, :, i), b[i])
    end

    BLAS.set_num_threads(prev_blas_threads)
end






"""CPU distributed-worker overload of `clenshaw` for [`TwoLevelParallel`](@ref)."""
function clenshaw(
    transform::ClenshawType,
    Y::AbstractVecOrMat{<:Number},
    X::AbstractVecOrMat{<:Number},
    b::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}},
    ::CPU,
    parallel_strategy::TwoLevelParallel,
)
    nvecs = size(X, 2)
    @assert size(X) == size(Y) "Y and Ỹ must be of the same size!"
    @assert length(b) == nvecs "Length of b must be the same as the number of columns of Y!"

    Y_sh = get_clenshaw_shared_buffer!(parallel_strategy, eltype(Y), size(Y))
    pmap(i -> transform(view(Y_sh, :, i), view(X, :, i), b[i]), parallel_strategy.worker_pool, 1:nvecs)
    Y .= Y_sh
end





"""
    get_b_storage(parallel::Parallelization, pu::ProcessingUnit, x0::AbstractVecOrMat)

Allocate Clenshaw recurrence workspace buffers.

Returned storage layout depends on strategy:
- `NoParallel`: 3 buffers matching `x0`.
- `MulColsParallel`/`TwoLevelParallel`: per-column triplets of vectors.
"""
function get_b_storage(parallel::Parallelization, pu::ProcessingUnit, x0::AbstractVecOrMat)

    E = eltype(x0)
    hilbertspacedim = size(x0,1)
    s = size(x0, 2)

    isa(parallel, NoParallel)       && (return [similar(x0) for _ in 1:3])
    isa(parallel, MulColsParallel)  && (return [[pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3] for _ in 1:s])
    isa(parallel, TwoLevelParallel) && (return [[pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3] for _ in 1:s])
end
