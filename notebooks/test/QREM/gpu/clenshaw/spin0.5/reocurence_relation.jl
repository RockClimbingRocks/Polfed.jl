

function clenshaw_reocurence_relation_vec_kernel_spin_onehalf(b1::CuDeviceVector, b2::CuDeviceVector, b3::CuDeviceVector, c::Real, X::CuDeviceVector, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    @inbounds @fastmath if i <= basis_length 
        d = diags[i]
        x = b2[i]

        offdiag_val = 0.0 
        @simd for j in 0:Leff
            newstate = (i-1) ⊻ (1 << j) 
            row = newstate + 1 
            offdiag_val += b2[row]
        end     
        yi = d*x + offdiag_val*hx

        b1[i] = c*X[i] + 2*yi - b3[i]
    end

    nothing
end


function clenshaw_reocurence_relation_mat_kernel_1d_spin_onehalf(b1::CuDeviceMatrix{T}, b2::CuDeviceMatrix{T}, b3::CuDeviceMatrix{T}, c::Real, X::CuDeviceMatrix{T}, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64, num_matrix_elements::Int64) where {T<:Float64}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    @inbounds @fastmath if i <= num_matrix_elements 
        i_rescaled = basis_length * iszero(i % basis_length) + i % basis_length
        d = diags[i_rescaled]
        x = b2[i]

        offdiag_val = 0.0
        @simd for l in 0:Leff
            newstate = (i-1) ⊻ (1 << l)
            row = newstate + 1 
            offdiag_val += b2[row]
        end     
        yi = d*x + offdiag_val*hx

        b1[i] = c*X[i] + 2*yi - b3[i]
    end

    nothing
end


function clenshaw_reocurence_relation_mat_kernel_2d_spin_onehalf(b1::CuDeviceMatrix, b2::CuDeviceMatrix, b3::CuDeviceMatrix, c::Real, X::CuDeviceMatrix, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    j = threadIdx().y

    @inbounds @fastmath if i <= basis_length 
        d = diags[i]
        x = b2[i,j]

        offdiag_val = 0.0 
        @simd for l in 0:Leff
            newstate = (i-1) ⊻ (1 << l)
            row = newstate + 1 
            offdiag_val += b2[row,j]
        end     
        yi = d*x + offdiag_val*hx

        b1[i,j] = c*X[i,j] + 2*yi - b3[i,j]
    end

    nothing
end