


@inline function mapping_state_i(
    X::AbstractVector{T}, 
    i::Int, 
    diagonals::AbstractVector, 
    val::Real, 
    offdiags_flatten::Vector{Int}, 
    start_indices::Vector{Int}
) where {T<:Real}

    start = start_indices[i]
    @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
    sum_val = T(0.0)
    for j in start:stop
        @inbounds sum_val += X[offdiags_flatten[j]]
    end

    @inbounds Y_i = muladd(diagonals[i], X[i], val * sum_val)
    return Y_i 
end



@inline function mapping_offdiagonals_state_i(
    X::AbstractVector{T}, 
    i::Int, 
    val::Real, 
    offdiags_flatten::Vector{Int}, 
    start_indices::Vector{Int}
) where {T<:Real}
    start = start_indices[i]
    @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
    sum_val = T(0.0)
    for j in start:stop
        @inbounds sum_val += X[offdiags_flatten[j]]
    end

    Y_off_val_i = val * sum_val
    return Y_off_val_i
end




