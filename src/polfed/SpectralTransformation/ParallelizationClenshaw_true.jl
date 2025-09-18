
const ClenshawType = Union{Clenshaw, ClenshawKernel}

# abstract type Parallelization end
# mutable struct NoParallel<:Parallelization end
# mutable struct MulColsParallel<:Parallelization end


# function clenshaw(
#     transform::ClenshawType, 
#     Ỹ::AbstractVecOrMat{<:Number}, 
#     Y::AbstractVecOrMat{<:Number}, 
#     b::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}}, 
#     ::CPU, 
#     ::MulColsParallel
# )
#     @assert size(Y) == size(Ỹ) "Y and Ỹ must be of the same size!"
#     @assert length(b) == size(Y, 2) "Length of b must be the same as the number of columns of Y!"

#     ncols = size(Y, 2)  # Number of columns
#     chunk_size = ceil(Int, ncols / nthreads())  # Determine chunk size based on number of threads

#     # Save the current number of BLAS threads
#     prev_blas_threads = BLAS.get_num_threads()
#     # Set BLAS threads to one
#     BLAS.set_num_threads(1)
#     @threads for t in 1:nthreads()
#         start_col = (t - 1) * chunk_size + 1
#         end_col = min(t * chunk_size, ncols)
#         for i in start_col:end_col
#             Ỹi = view(Ỹ, :, i)
#             Yi = view(Y, :, i)
#             @views transform(Ỹi, Yi, b[i])
#         end
#     end
#     # Restore the previous number of BLAS threads
#     BLAS.set_num_threads(prev_blas_threads)
# end



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


function get_b_storage(parallel::Parallelization, pu::ProcessingUnit, x0::AbstractVecOrMat)

    E = eltype(x0)
    hilbertspacedim = size(x0,1)
    s = size(x0, 2)

    isa(parallel, MulColsParallel)  && (return [[pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3] for _ in 1:s])
    isa(parallel, NoParallel)       && (return [similar(x0) for _ in 1:3])
end

