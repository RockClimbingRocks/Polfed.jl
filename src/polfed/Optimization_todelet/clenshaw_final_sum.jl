
"""
    optimized_clenshaw_final_sum!(diagonals, offdiagonals)

Return a closure that performs the final Clenshaw summation step:
    Y = c * X + H * b1 - b2

where H is represented by diagonals + offdiagonals.
"""
function optimized_clenshaw_final_sum!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy::Parallelization)
    cfs = @inline (b1, b2, c, Y, X) -> begin
        map_with_clenshaw_final_sum!(b1, b2, c, Y, X, diagonals, offdiagonals)
    end
    return cfs
end





# -------------------------------
# Matrix version (each column treated like a vector)
# -------------------------------
function map_with_clenshaw_final_sum!(
    b1::AbstractMatrix{T}, 
    b2::AbstractMatrix{T}, 
    c::Real, 
    Y::AbstractMatrix{T}, 
    X::AbstractMatrix{T}, 
    diagonals::AbstractVector{T}, 
    offdiagonals::Union{
        Tuple{<:Number, Vector{Int}, Vector{Int}},
        Vector{Tuple{<:Number, Vector{Int}, Vector{Int}}}
    }
) where {T<:Number}

    ncols = size(b1, 2)
    
    for col in 1:ncols
        view_b1 = view(b1, :, col)
        view_b2 = view(b2, :, col)
        view_Y  = view(Y,  :, col)
        view_X  = view(X,  :, col)

        map_with_clenshaw_final_sum!(view_b1, view_b2, c, view_Y, view_X, diagonals, offdiagonals)
    end
end



# -------------------------------
# Vector version, single offdiagonal
# -------------------------------
function map_with_clenshaw_final_sum!(
    b1::AbstractVector{T}, 
    b2::AbstractVector{T}, 
    c::Real, 
    Y::AbstractVector{T}, 
    X::AbstractVector{T}, 
    diagonals::AbstractVector{T}, 
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}}
) where {T<:Number}

    (val, offdiags_flatten, start_indices) = offdiagonals

    for i in eachindex(start_indices)
        @inbounds start = start_indices[i]
        @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

        sum_val = T(0.0)
        for j in start:stop
            @inbounds sum_val += b1[offdiags_flatten[j]]
        end

        @inbounds yi = muladd(diagonals[i], b1[i], val * sum_val)
        @inbounds Y[i] = c*X[i] + yi - b2[i]
    end
end





# -------------------------------
# Vector version, single offdiagonal
# -------------------------------
function map_with_clenshaw_final_sum_parallel!(
    b1::AbstractVector{T}, 
    b2::AbstractVector{T}, 
    c::Real, 
    Y::AbstractVector{T}, 
    X::AbstractVector{T}, 
    diagonals::AbstractVector{T}, 
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}}
) where {T<:Number}

    (val, offdiags_flatten, start_indices) = offdiagonals

    @threads for i in eachindex(start_indices)
        @inbounds start = start_indices[i]
        @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

        sum_val = T(0.0)
        for j in start:stop
            @inbounds sum_val += b1[offdiags_flatten[j]]
        end

        @inbounds yi = muladd(diagonals[i], b1[i], val * sum_val)
        @inbounds Y[i] = c*X[i] + yi - b2[i]
    end
end





# -------------------------------
# Vector version, multiple offdiagonals
# -------------------------------
function map_with_clenshaw_final_sum!(
    b1::AbstractVector{T}, 
    b2::AbstractVector{T}, 
    c::Real, 
    Y::AbstractVector{T}, 
    X::AbstractVector{T}, 
    diagonals::AbstractVector{T}, 
    offdiagonals::Vector{Tuple{<:Number, Vector{Int}, Vector{Int}}}
) where {T<:Number}

    for i in eachindex(diagonals)
        # diagonal contribution
        yi = diagonals[i] * b1[i]

        # offdiagonal contributions
        for (val, flat, starts) in offdiagonals
            @inbounds start = starts[i]
            @inbounds stop  = (i == length(starts)) ? length(flat) : starts[i+1]-1

            sum_val = T(0.0)
            for j in start:stop
                @inbounds sum_val += b1[flat[j]]
            end
            yi += val * sum_val
        end

        @inbounds Y[i] = c*X[i] + yi - b2[i]
    end
end





# -------------------------------
# Vector version, multiple offdiagonals
# -------------------------------
function map_with_clenshaw_final_sum_parallel!(
    b1::AbstractVector{T}, 
    b2::AbstractVector{T}, 
    c::Real, 
    Y::AbstractVector{T}, 
    X::AbstractVector{T}, 
    diagonals::AbstractVector{T}, 
    offdiagonals::Vector{Tuple{<:Number, Vector{Int}, Vector{Int}}}
) where {T<:Number}

    @threads for i in eachindex(diagonals)
        # diagonal contribution
        yi = diagonals[i] * b1[i]

        # offdiagonal contributions
        for (val, flat, starts) in offdiagonals
            @inbounds start = starts[i]
            @inbounds stop  = (i == length(starts)) ? length(flat) : starts[i+1]-1

            sum_val = T(0.0)
            for j in start:stop
                @inbounds sum_val += b1[flat[j]]
            end
            yi += val * sum_val
        end

        @inbounds Y[i] = c*X[i] + yi - b2[i]
    end
end

