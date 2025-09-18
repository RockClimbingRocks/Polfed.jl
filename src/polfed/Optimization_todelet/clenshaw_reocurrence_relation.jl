

function optimized_clenshaw_recurrence_relation!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy::Parallelization)
    
    crr = nothing

    if parallel_strategy isa NoParallel
        crr = (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, X::AbstractVecOrMat) -> begin
            map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals)
        end
    elseif parallel_strategy isa MulColsParallel
        crr = (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, X::AbstractVecOrMat) -> begin
            map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals)
        end

    elseif parallel_strategy isa TwoLevelParallel
        crr = (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, X::AbstractVecOrMat) -> begin
            map_with_clenshaw_recurrence_relation_parallel!(b1, b2, b3, c, X, diagonals, offdiagonals)
        end
    else
        error("Unknown parallelization strategy: $parallel_strategy")
    end

    return crr
end


"""
    map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals)

Matrix generalization: applies the Clenshaw recurrence column-wise.
Works for both vectors and matrices.
"""
function map_with_clenshaw_recurrence_relation!(
    b1::AbstractMatrix{T}, 
    b2::AbstractMatrix{T}, 
    b3::AbstractMatrix{T}, 
    c::Real, 
    X::AbstractMatrix{T},
    diagonals::AbstractVector, 
    offdiagonals::Offdiagonals
) where {T<:Number}

    @assert size(b1) == size(b2) == size(b3) == size(X)

    # Process each column independently
    for col in axes(X, 2)
        map_with_clenshaw_recurrence_relation!(
            view(b1, :, col), 
            view(b2, :, col), 
            view(b3, :, col), 
            c, 
            view(X, :, col),
            diagonals, 
            offdiagonals
        )
    end

    return b1
end


function map_with_clenshaw_recurrence_relation!(
    b1::AbstractVecOrMat{T}, 
    b2::AbstractVecOrMat{T}, 
    b3::AbstractVecOrMat{T}, 
    c::Real, 
    X::AbstractVecOrMat{T},
    diagonals::AbstractVector, 
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
) where {T<:Number}
    (val, offdiags_flatten, start_indices) = offdiagonals


    for i in eachindex(start_indices)
        @inbounds start = start_indices[i]
        @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

        sum_val = T(0.0)
        for j in start:stop
            @inbounds sum_val += b2[offdiags_flatten[j]]
        end

        @inbounds yi = diagonals[i] * b2[i] + val * sum_val
        @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
    end
end



function map_with_clenshaw_recurrence_relation!(
    b1::AbstractVecOrMat{T}, 
    b2::AbstractVecOrMat{T}, 
    b3::AbstractVecOrMat{T}, 
    c::Real, 
    X::AbstractVecOrMat{T},
    diagonals::AbstractVector, 
    offdiagonals::Vector{Tuple{<:Number, Vector{Int}, Vector{Int}}},
) where {T<:Number}

    for i in eachindex(diagonals)
        # Start with the diagonal part
        @inbounds yi = diagonals[i] * b2[i]

        # Add contributions from each off-diagonal group
        for (val, flat, starts) in offdiagonals
            @inbounds start = starts[i]
            @inbounds stop  = (i == length(starts)) ? length(flat) : starts[i+1]-1

            sum_val = T(0.0)
            for j in start:stop
                @inbounds sum_val += b2[flat[j]]
            end
            yi += val * sum_val
        end

        # Finish the Clenshaw step
        @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
    end
end





function map_with_clenshaw_recurrence_relation_parallel!(
    b1::AbstractVecOrMat{T}, 
    b2::AbstractVecOrMat{T}, 
    b3::AbstractVecOrMat{T}, 
    c::Real, 
    X::AbstractVecOrMat{T},
    diagonals::AbstractVector, 
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
) where {T<:Number}
    (val, offdiags_flatten, start_indices) = offdiagonals


    @threads for i in eachindex(start_indices)
        @inbounds start = start_indices[i]
        @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

        sum_val = T(0.0)
        for j in start:stop
            @inbounds sum_val += b2[offdiags_flatten[j]]
        end

        @inbounds yi = diagonals[i] * b2[i] + val * sum_val
        @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
    end
end



function map_with_clenshaw_recurrence_relation_parallel!(
    b1::AbstractVecOrMat{T}, 
    b2::AbstractVecOrMat{T}, 
    b3::AbstractVecOrMat{T}, 
    c::Real, 
    X::AbstractVecOrMat{T},
    diagonals::AbstractVector, 
    offdiagonals::Vector{Tuple{<:Number, Vector{Int}, Vector{Int}}},
) where {T<:Number}

    @threads for i in eachindex(diagonals)
        # Start with the diagonal part
        @inbounds yi = diagonals[i] * b2[i]

        # Add contributions from each off-diagonal group
        for (val, flat, starts) in offdiagonals
            @inbounds start = starts[i]
            @inbounds stop  = (i == length(starts)) ? length(flat) : starts[i+1]-1

            sum_val = T(0.0)
            for j in start:stop
                @inbounds sum_val += b2[flat[j]]
            end
            yi += val * sum_val
        end

        # Finish the Clenshaw step
        @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
    end
end
