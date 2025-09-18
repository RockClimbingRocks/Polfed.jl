
const ClenshawType = Union{Clenshaw, ClenshawKernel}





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
    Y::AbstractVecOrMat{<:Number},
    X::AbstractVecOrMat{<:Number},
    b::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}},
    ::CPU,
    parallelization::TwoLevelParallel, # Now passing the whole struct
)
    nvecs = size(X, 2)
    @assert size(X) == size(Y) "Y and Ỹ must be of the same size!"
    @assert length(b) == nvecs "Length of b must be the same as the number of columns of Y!"


    Y_sh = SharedArray{eltype(Y)}(size(Y))
    pmap(i -> transform(view(Y_sh, :, i), view(X, :, i), b[i]), 1:nvecs)
    Y .= Y_sh
end





function get_b_storage(parallel::Parallelization, pu::ProcessingUnit, x0::AbstractVecOrMat)

    E = eltype(x0)
    hilbertspacedim = size(x0,1)
    s = size(x0, 2)

    isa(parallel, NoParallel)       && (return [similar(x0) for _ in 1:3])
    isa(parallel, MulColsParallel)  && (return [[pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3] for _ in 1:s])
    isa(parallel, TwoLevelParallel) && (return [[pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3] for _ in 1:s])
end

