
# # Here i should write function that parallizes mul mapping!#
# # So that each column could have multiple threads!

"""
    parallelize_mul_per_col!(mat::AbstractVecOrMat, parallel_strategy::Parallelization) -> Function

Return an in-place multiplication callback `f!(Y, X)` according to the selected
parallelization strategy.

- GPU arrays and `NoParallel` use plain `mul!`.
- `MulColsParallel` can optionally split row work per column when
  `nt_per_col > 1`.
"""
function parallelize_mul_per_col!(mat::AbstractVecOrMat, parallel_strategy::Parallelization)

    f! = nothing
    if is_gpu_array(mat) || parallel_strategy isa NoParallel
        f! = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> mul!(Y , mat, X)
    elseif parallel_strategy isa MulColsParallel
        nt_per_col = parallel_strategy.nt_per_col

        if nt_per_col==1
            f! = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> mul!(Y , mat, X)
        else
            f! = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
                parallelize_mul_per_col!(Y, mat, X, nt_per_col) 
            
                # if Y isa AbstractVector && X isa AbstractVector
                #     parallelize_mul_per_col_vec!(Y, mat, X, nt_per_col)
                # elseif Y isa AbstractMatrix && X isa AbstractMatrix
                #     # println("implement this!!! think if it is useful!!!")
                #     # throw(ArgumentError("Parallelization for matrix multiplication not implemented yet!"))
                #     # parallelize_mul_per_col_mat!(Y, mat, X, nt_per_col)
                # end 
            end
        end
    end


    return f!
end

"""
    parallelize_mul_per_col!(Y::AbstractVector, mat::AbstractMatrix, X::AbstractVector, nt_per_col::Int) -> nothing

Compute `Y = mat * X` by splitting output rows across `nt_per_col` threads.
`Y` is mutated in-place.
"""
function parallelize_mul_per_col!(Y::AbstractVector, mat::AbstractMatrix, X::AbstractVector, nt_per_col::Int)
    len = length(Y)

    if len == 0 || nt_per_col <= 1
        mul!(Y, mat, X)
        return
    end

    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)  # avoid BLAS multithreading inside our threads

    chunk_size = cld(len, nt_per_col)

    Threads.@threads for i in 1:nt_per_col
        start_idx = (i - 1) * chunk_size + 1
        end_idx   = min(i * chunk_size, len)

        if start_idx <= len
            Yi   = view(Y, start_idx:end_idx)
            mati = view(mat, start_idx:end_idx, :)
            mul!(Yi, mati, X)
        end
    end

    BLAS.set_num_threads(prev_blas_threads)
end

"""
    parallelize_mul_per_col!(Y::AbstractMatrix, mat::AbstractMatrix, X::AbstractMatrix, nt_per_col::Int) -> nothing

Compute `Y = mat * X` by splitting output row blocks across `nt_per_col`
threads. `Y` is mutated in-place.
"""
function parallelize_mul_per_col!(Y::AbstractMatrix, mat::AbstractMatrix, X::AbstractMatrix, nt_per_col::Int)
    nrows = size(Y, 1)

    if nrows == 0 || nt_per_col <= 1
        mul!(Y, mat, X)
        return
    end

    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)  # avoid BLAS multithreading inside threads

    chunk_size = cld(nrows, nt_per_col)

    Threads.@threads for i in 1:nt_per_col
        start_row = (i - 1) * chunk_size + 1
        end_row   = min(i * chunk_size, nrows)

        if start_row <= nrows
            Yi   = view(Y, start_row:end_row, :)
            mati = view(mat, start_row:end_row, :)
            mul!(Yi, mati, X)
        end
    end

    BLAS.set_num_threads(prev_blas_threads)
end
# function parallelize_mul_per_col!(Y::AbstractMatrix, mat::AbstractMatrix, X::AbstractMatrix, nt_per_col::Int)

#     len = length(Y)
#     if len == 0 || nt_per_col <= 1
#         mul!(Y, mat, X)
#         return
#     end

#     @sync begin
#         for i in 1:nt_per_col
#             chunk_size = cld(len, nt_per_col) 
#             start_idx = (i - 1) * chunk_size + 1
#             end_idx = min(i * chunk_size, len)
            
#             if start_idx > len
#                 break
#             end

#             Threads.@spawn begin
#                 Yi = view(Y, start_idx:end_idx, :)
#                 mati = view(mat, start_idx:end_idx, :)
#                 mul!(Yi, mati, X)
#             end
#         end
#     end     
# end








# function parallelize_mul_per_col_vec!(
#     diags::Vector{Float64},
#     offdiags_flatten::Vector{Int},
#     start_indices::Vector{Int},
#     J::Float64,
#     nt_per_col::Integer
# )
#     J_half = J / 2
#     len = length(start_indices)

#     return (Y,X) -> @sync begin
#         for tid in 1:nt_per_col
#             chunk_size = cld(len, nt_per_col) 
#             start_idx = (tid - 1) * chunk_size + 1
#             end_idx = min(tid * chunk_size, len)
            
#             start_idx > len && (break)

#             Threads.@spawn begin
#                 for row in start_idx:end_idx
#                     @inbounds start = start_indices[row]
#                     @inbounds stop  = (row == len) ? length(offdiags_flatten) : start_indices[row+1]-1
#                     sum_val = 0.0
#                     for j in start:stop
#                         @inbounds sum_val += X[offdiags_flatten[j]]
#                     end
#                     @inbounds Y[row] = muladd(diags[row], X[row], J_half * sum_val)
#                 end
#             end
#         end
#     end     
# end

# function mapvec_with_xxz!(
#     Y::AbstractVector,
#     X::AbstractVector,
#     diags::Vector{Float64},
#     offdiags_flatten::Vector{Int},
#     start_indices::Vector{Int},
#     J::Float64
# )

#     J_half =J/2
#     for i in eachindex(start_indices)
#         @inbounds start = start_indices[i]
#         @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
#         sum_val = 0.0
#         for j in start:stop
#             @inbounds sum_val += X[offdiags_flatten[j]]
#         end
#         @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
#     end
# end



