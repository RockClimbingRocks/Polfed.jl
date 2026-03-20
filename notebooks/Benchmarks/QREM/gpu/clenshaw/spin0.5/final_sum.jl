
function clenshaw_finalsum_vec_kernel_spin_onehalf(b1::CuDeviceVector, b2::CuDeviceVector, c::Real, Y::CuDeviceVector, X::CuDeviceVector, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    @inbounds @fastmath if i <= basis_length 
        d = diags[i]
        x = b1[i]
        # X here goes to b1

        offdiag_val = 0.0 
        @simd for j in 0:Leff
            newstate = (i-1) ⊻ (1 << j) 
            row = newstate + 1 
            offdiag_val += b1[row] 
        end     
        y = d*x + offdiag_val*hx

        Y[i] = c*X[i] + y - b2[i]
    end

    nothing
end


function clenshaw_finalsum_mat_kernel_1d_spin_onehalf(b1::CuDeviceMatrix, b2::CuDeviceMatrix, c::Real, Y::CuDeviceMatrix, X::CuDeviceMatrix, hx::Float64, diags::CuDeviceVector{T}, Leff::Int64, basis_length::Int64, matrixelements::Int64) where {T<:Float64}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    @inbounds @fastmath if i <= matrixelements 
        i_rescaled = basis_length * iszero(i % basis_length) + i % basis_length
        d = diags[i_rescaled] # "load" diag. elem.
        x = b1[i]
        # X here goes to b1

        offdiag_val = 0.0 
        @simd for j in 0:Leff
            newstate = (i-1) ⊻ (1 << j) 
            row = newstate + 1 
            offdiag_val += b1[row] 
        end     
        y = d*x + offdiag_val*hx

        Y[i] = c*X[i] + y - b2[i]
    end

    nothing
end


function tfim_finalsum_mat_kernel_2d_spin_onehalf(b1::CuDeviceMatrix, b2::CuDeviceMatrix, c::Real, Y::CuDeviceMatrix, X::CuDeviceMatrix, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    j = threadIdx().y

    @inbounds @fastmath if i <= basis_length 
        d = diags[i]
        x = b1[i,j]
        # X here goes to b1

        offdiag_val = 0.0 
        @simd for l in 0:Leff
            newstate = (i-1) ⊻ (1 << l) 
            row = newstate + 1 
            offdiag_val += b1[row,j] 
        end     
        y = d*x + offdiag_val*hx

        Y[i,j] = c*X[i,j] + y - b2[i,j]
    end

    nothing
end