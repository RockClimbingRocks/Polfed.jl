
CUDA.@device_function function mapping_state_i!(
    X::CuDeviceVector{T},
    i::Int,
    D::CuDeviceVector{T},
    val::Real,
    flat::CuDeviceVector{Int},
    starts::CuDeviceVector{Int},
    N::Int
) where {T<:Real}

    start = starts[i]
    stop = (i == length(starts)) ? N : starts[i+1] - 1

    sum_val = T(0.0)
    for j in start:stop
        sum_val += X[flat[j]]
    end

    return D[i] * X[i] + val * sum_val
end


CUDA.@device_function function mapping_state_i!(
    X::CuDeviceMatrix{T},
    i::Int,
    i_rescaled::Int,
    D::CuDeviceVector{T},
    val::Real,
    flat::CuDeviceVector{Int},
    starts::CuDeviceVector{Int},
    N::Int
) where {T<:Real}

    start = starts[i_rescaled]
    stop = (i_rescaled == length(starts)) ? N : starts[i_rescaled+1] - 1

    sum_val = T(0.0)
    for j in start:stop
        sum_val += X[flat[j]]
    end

    return val * sum_val
end


# CUDA.@device_function function mapping_state_i!(
#     X::CuDeviceMatrix{T},
#     i::Int,
#     j::Int,
#     D::CuDeviceMatrix{T},
#     val::Real,
#     flat::CuDeviceVector{Int},
#     starts::CuDeviceVector{Int},
#     N::Int
# ) where {T<:Real}

#     start = starts[i]
#     stop = (i == length(starts)) ? N : starts[i+1] - 1

#     sum_val = T(0.0)
#     for k in start:stop
#         sum_val += X[flat[k],j]
#     end
    

#     return D[i,k] * X[i,k] + val * sum_val
# end












CUDA.@device_function function mapping_state_i_offdiag!(
    X::CuDeviceVector{T},
    i::Int,
    val::Real,
    flat::CuDeviceVector{Int},
    starts::CuDeviceVector{Int},
    N::Int
) where {T<:Real}

    start = starts[i]
    stop = (i == length(starts)) ? N : starts[i+1] - 1

    sum_val = T(0.0)
    for j in start:stop
        sum_val += X[flat[j]]
    end

    return val * sum_val
end




CUDA.@device_function function mapping_state_i_offdiag!(
    X::CuDeviceMatrix{T},
    i::Int,
    i_rescaled::Int,
    val::Real,
    flat::CuDeviceVector{Int},
    starts::CuDeviceVector{Int},
    N::Int
) where {T<:Real}

    start = starts[i_rescaled]
    stop = (i == length(starts)) ? N : starts[i_rescaled+1] - 1

    sum_val = T(0.0)
    for j in start:stop
        sum_val += X[flat[j]]
    end

    return val * sum_val
end




