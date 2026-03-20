"""
    tfim_map_mat_kernel_1d(Y::CuDeviceMatrix{T}, X::CuDeviceMatrix{T}, Leff::Int, diag::CuDeviceVector{T}, hx::T, basis_length::Int, matrixelements::Int) where {T<:Float64}

This function performs a kernel operation for mapping a 2D matrix in the context of the Transverse Field Ising Model (TFIM) on a GPU, kernel uses ONE dimensional grid. It computes the result of applying the Hamiltonian matrix to a given input matrix `X` and stores the result in the output matrix `Y`.

# Arguments
- `Y::CuDeviceMatrix{T}`: Output matrix to store the result of the transformation.
- `X::CuDeviceMatrix{T}`: Input matrix representing the state vector.
- `Leff::Int`: Effective system size, representing the number of spins.
- `diag::CuDeviceVector{T}`: Diagonal elements of the sparse matrix.
- `hx::T`: Transverse field strength.
- `basis_length::Int`: Length of the basis vector.
- `matrixelements::Int`: Total number of matrix elements to process.

# Details
- The diagonal contribution is computed using the `diag` vector.
- The off-diagonal contribution is computed by flipping individual bits of the state index using the σ_x operator.
- The result is stored in the output matrix `Y`.

# Notes
- This function is designed to run on CUDA-enabled GPUs.
- The computation assumes 1-based indexing for the state vector.
- The `@inbounds` and `@fastmath` macros are used to optimize performance by skipping bounds checking and enabling fast math operations.

# Returns
- `nothing`: The result is stored directly in the `Y` matrix.
"""
function tfim_map_mat_kernel_1d(Y::CuDeviceMatrix{T}, X::CuDeviceMatrix{T}, Leff::Int, diag::CuDeviceVector{T}, hx::T, basis_length::Int, matrixelements::Int) where {T<:Float64}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds @fastmath if i <= matrixelements
        i_rescaled = basis_length * iszero(i % basis_length) + i % basis_length
        d = diag[i_rescaled] # "load" diag. elem.
        x = X[i]
        
        offdiag_val = 0.0 #where to store the sum of offdiagonals 
        @simd for l in 0:Leff
            newstate = (i-1) ⊻ (1 << l) # σ_x operator: Flip the j-th bit
            row = newstate + 1 # because of 1-based enumeration of vectors
            offdiag_val += X[row] 
        end     
        
        Y[i] = d*x + offdiag_val*hx #store the result to finial array
    end

    nothing
end


"""
    tfim_map_mat_kernel_2d(Y::CuDeviceMatrix{T}, X::CuDeviceMatrix{T}, Leff::Int, diag::CuDeviceVector{T}, hx::T, basis_length::Int) where {T<:Real}

This function performs a kernel operation for mapping a 2D matrix in the context of the Transverse Field Ising Model (TFIM) on a GPU, kernel uses TWO dimensional grid. It computes the result of applying the Hamiltonian matrix to a given input matrix `X` and stores the result in the output matrix `Y`.

# Arguments
- `Y::CuDeviceMatrix{T}`: The output matrix where the result of the operation will be stored.
- `X::CuDeviceMatrix{T}`: The input matrix representing the current state.
- `Leff::Int`: The effective system size, representing the number of spins in the system.
- `diag::CuDeviceVector{T}`: A vector containing the diagonal elements of the Hamiltonian matrix.
- `hx::T`: The transverse field strength, a scalar value.
- `basis_length::Int`: The length of the basis, representing the number of basis states.

# Details
- The function uses GPU threads to parallelize the computation. Each thread computes a specific element of the output matrix `Y`.
- The diagonal contribution is computed using the `diag` vector and the corresponding element of `X`.
- The off-diagonal contribution is computed by flipping each bit in the binary representation of the current state index `i` using the σₓ operator and summing the contributions from the corresponding rows of `X`.
- The final result for each element of `Y` is the sum of the diagonal and off-diagonal contributions, scaled by the transverse field strength `hx`.

# Notes
- The function assumes 1-based indexing for the input and output matrices, consistent with Julia's indexing.
- The `@inbounds` and `@fastmath` macros are used to optimize performance by skipping bounds checking and enabling fast math operations.
- The `@simd` macro is used to enable SIMD (Single Instruction, Multiple Data) optimizations for the loop over `l`.

# Returns
- The function does not return a value explicitly. The result is stored directly in the `Y` matrix.
"""



function tfim_map_mat_kernel_2d(Y::CuDeviceMatrix{T}, X::CuDeviceMatrix{T}, Leff::Int, diag::CuDeviceVector{T}, hx::T, basis_length::Int) where {T<:Real}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = threadIdx().y

    @inbounds @fastmath if i <= basis_length
        d = diag[i] # "load" diag. elem.
        x = X[i,j]
        
        offdiag_val = 0.0 #where to store the sum of offdiagonals 
        @simd for l in 0:Leff
            newstate = (i-1) ⊻ (1 << l) # σ_x operator: Flip the j-th bit
            row = newstate + 1 # because of 1-based enumeration of vectors
            offdiag_val += X[row, j] 
        end     
        
        Y[i, j] = d*x + offdiag_val*hx #store the result to finial array
    end

    nothing
end

