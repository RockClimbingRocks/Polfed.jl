"""
    tfim_map_kernel(Y::CuDeviceVector{T}, X::CuDeviceVector{T}, Leff::Int, diag::CuDeviceVector{T}, hx::Real, basis_length::Int) where {T<:Real}

Performs a kernel operation for the Transverse Field Ising Model (TFIM) on GPU. This function maps the input vector `X` to the output vector `Y` using the specified parameters.

# Arguments
- `Y::CuDeviceVector{T}`: The output vector where the result of the operation will be stored.
- `X::CuDeviceVector{T}`: The input vector to be processed.
- `Leff::Int`: The effective system size, representing the number of spins in the system.
- `diag::CuDeviceVector`: A vector containing the diagonal elements of the Hamiltonian.
- `hx::Real`: The transverse field strength.
- `basis_length::Int`: The length of the basis, which determines the size of the input and output vectors.

# Details
- The function computes the result for each index `i` in the range `[1, basis_length]` using GPU threads.
- For each index `i`, the function calculates:
  - The diagonal contribution as `diag[i] * X[i]`.
  - The off-diagonal contribution by iterating over all possible spin flips (determined by `Leff`) and summing the corresponding values from `X`.
- The final result for each index is stored in `Y[i]` as the sum of the diagonal and off-diagonal contributions, scaled by the transverse field `hx`.

# Notes
- The function uses `@inbounds` to skip bounds checking for performance.
- The `@fastmath` macro is used to enable fast math optimizations.
- The `@simd` macro is used to enable SIMD (Single Instruction, Multiple Data) optimizations for the loop over spin flips.
- The function assumes 1-based indexing for vectors, consistent with Julia's indexing.

# GPU Considerations
- The function is designed to be executed on a GPU using CUDA. Each thread computes the result for a single index `i`.
- The thread index `i` is calculated based on the block and thread dimensions.

# Returns
- The function does not return a value. The result is stored directly in the `Y` vector.
"""

function qrem_map_kernel_spin_half(Y::CuDeviceVector{T}, X::CuDeviceVector{T}, Leff::Int, diag::CuDeviceVector, hx::Real, basis_length::Int) where {T<:Real}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds @fastmath if i <= basis_length # Ensures the thread index does not exceed the vector size to prevent out-of-bounds access
        d = diag[i] 
        x = X[i]

        offdiag_val = 0.0 

        # might be faster storing all rows and then acces the memory at once
        @simd for l in 0:Leff
            newstate = (i-1) ⊻ (1 << l) 
            row = newstate + 1 # because of 1-based enumeration of vectors
            offdiag_val += X[row] 
        end     
        
        Y[i] = d*x + offdiag_val*hx #store the result to finial array
    end

    nothing
end






