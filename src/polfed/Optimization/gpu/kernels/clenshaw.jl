



#---------------------------------------------------
# CASE 1: Vector Inputs, Single Off-diagonal
#---------------------------------------------------
function clenshaw_recurrence_one_offdiag!(
    b1::CuDeviceVector{T}, 
    b2::CuDeviceVector{T}, 
    b3::CuDeviceVector{T}, 
    c::Real,
    X::CuDeviceVector{T}, 
    D::CuDeviceVector{T}, 
    val::Real, 
    flat::CuDeviceVector{Int}, 
    starts::CuDeviceVector{Int}) where {T<:Real}

    # Each thread calculates its own unique index 'i'
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    N = length(X)
    
    # Boundary check: ensures we don't try to access memory out of bounds
    if i <= N
        Yi = mapping_state_i(b2, i, D, val, flat, starts, N)
        @inbounds b1[i] = c*X[i] + 2*Yi - b3[i]
    end

    return nothing
end



function clenshaw_recurrence_one_offdiag!(
    b1::CuDeviceMatrix{T}, 
    b2::CuDeviceMatrix{T}, 
    b3::CuDeviceMatrix{T}, 
    c::Real, 
    X::CuDeviceMatrix{T},
    D::CuDeviceVector{T}, 
    val::Real, 
    flat::CuDeviceVector{Int}, 
    starts::CuDeviceVector{Int}) where {T<:Real}

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    (N, blocksize) = size(X, 1)
    num_matrix_elements = N * blocksize


    @inbounds if i <= num_matrix_elements 
        i_rescaled = basis_length * iszero(i % basis_length) + i % basis_length

        Yi_offdiag = mapping_state_i_offdiag!(b2, i, val, flat, starts, N)
        yi = D[i_rescaled]*b2[i] + Yi_offdiag

        b1[i] = c*X[i] + 2*yi - b3[i]
    end

    nothing
end


# function clenshaw_recurrence_one_offdiag!(
#     b1::CuDeviceMatrix{T}, 
#     b2::CuDeviceMatrix{T}, 
#     b3::CuDeviceMatrix{T}, 
#     c::Real, 
#     X::CuDeviceMatrix{T},
#     D::CuDeviceVector{T},
#     val::Real,
#     flat::CuDeviceVector{Int},
#     starts::CuDeviceVector{Int}) where {T<:Real}

#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
#     j = threadIdx().y

#     (basis_length, blocksize) = size(X, 1)

#     @inbounds @fastmath if i <= basis_length && j <= blocksize

#         Yij = mapping_state_i!(b2, i, j, D, val, flat, starts, basis_length)
#         b1[i,j] = c*X[i,j] + 2*Yij - b3[i,j]
#     end

#     nothing
# end

