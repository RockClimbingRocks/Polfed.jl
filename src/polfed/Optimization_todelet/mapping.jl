
function optimized_mapping!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy::Parallelization)
    mapping_ = nothing 



    if parallel_strategy isa NoParallel
        mapping_ = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
            @assert size(X) == size(Y)
            mapping!(Y, X, diagonals, offdiagonals)
        end
    elseif parallel_strategy isa MulColsParallel
        mapping_ = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
            @assert size(X) == size(Y)
            mapping!(Y, X, diagonals, offdiagonals)
        end
    elseif parallel_strategy isa TwoLevelParallel
        mapping_ = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
            @assert size(X) == size(Y)
            mapping_parallel!(Y, X, diagonals, offdiagonals)
        end
    else
        error("Unknown parallelization strategy: $parallel_strategy")
    end

    return mapping_
end

function mapping!(
    Y::AbstractMatrix, 
    X::AbstractMatrix, 
    diagonals::AbstractVector{T},
    offdiagonals::Offdiagonal,
) where {T<:Real}
    @assert size(Y) == size(X)
    for col in axes(X,2)
        mapping!(view(Y,:,col), view(X,:,col), diagonals, offdiagonals)
    end
end


function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector{T},
    offdiagonals::Tuple{T, Vector{Int}, Vector{Int}},
) where {T<:Real}

    (val, offdiags_flatten, start_indices) = offdiagonals
    
    for i in eachindex(start_indices)
        start = start_indices[i]
        @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
        sum_val = T(0.0)
        for j in start:stop
            @inbounds sum_val += X[offdiags_flatten[j]]
        end
        @inbounds Y[i] = muladd(diagonals[i], X[i], val * sum_val) 
    end
end



function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector{T},
    offdiagonals::Vector{Tuple{T, Vector{Int}, Vector{Int}}},
) where {T<:Real}

    @. Y = diagonals * X
    
    for (val, offdiags_flatten, start_indices) in offdiagonals
        for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = T(0.0)
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] += sum_val * val
        end
    end
end



function mapping_parallel!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector{T},
    offdiagonals::Tuple{T, Vector{Int}, Vector{Int}},
) where {T<:Real}
    (val, offdiags_flatten, start_indices) = offdiagonals
    
    @threads for i in eachindex(start_indices)
        start = start_indices[i]
        @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
        sum_val = T(0.0)
        for j in start:stop
            @inbounds sum_val += X[offdiags_flatten[j]]
        end
        @inbounds Y[i] = muladd(diagonals[i], X[i], val * sum_val) 
    end
end



function mapping_parallel!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector{T},
    offdiagonals::Vector{Tuple{T, Vector{Int}, Vector{Int}}},
) where {T<:Real}

    @. Y = diagonals * X
    
    for (val, offdiags_flatten, start_indices) in offdiagonals
        @threads for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = T(0.0)
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] += sum_val * val
        end
    end
end

