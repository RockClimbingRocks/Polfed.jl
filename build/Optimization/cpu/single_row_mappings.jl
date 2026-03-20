


"""
    mapping_state_i(X, i, diagonals, val, offdiags_flatten, start_indices)

Compute one row output value for a packed single-value off-diagonal mapping.
"""
@inline function mapping_state_i(
    X::AbstractVector{T}, 
    i::Int, 
    diagonals::AbstractVector, 
    val::Number, 
    offdiags_flatten::Vector{Int}, 
    start_indices::Vector{Int}
) where {T<:Number}

    start = start_indices[i]
    @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
    sum_val = zero(T)
    for j in start:stop
        @inbounds sum_val += X[offdiags_flatten[j]]
    end

    @inbounds Y_i = diagonals[i] * X[i] + val * sum_val
    return Y_i 
end



"""
    mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)

Compute only the off-diagonal contribution for one row and one value bucket.
"""
@inline function mapping_offdiagonals_state_i(
    X::AbstractVector{T}, 
    i::Int, 
    val::Number, 
    offdiags_flatten::Vector{Int}, 
    start_indices::Vector{Int}
) where {T<:Number}
    start = start_indices[i]
    @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
    sum_val = zero(T)
    for j in start:stop
        @inbounds sum_val += X[offdiags_flatten[j]]
    end

    Y_off_val_i = val * sum_val
    return Y_off_val_i
end

