
using SparseArrays, LinearAlgebra, BenchmarkTools, Base.Threads
include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
# using .Polfed


function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
    basis = [b for b in 0:2^L-1 if count_ones(b) == Nup] # generate basis
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))  # state index map
    rows, cols, vals = Int[], Int[], Float64[]
    for (col, state) in enumerate(basis)
        for i in 1:L
            j = i % L + 1  # PBC
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1
            SzSz = (0.5 - si) * (0.5 - sj) 
            push!(rows, col); push!(cols, col); push!(vals, delta * SzSz)
            if si != sj
                flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                if haskey(bmap, flipped) 
                    push!(rows, bmap[flipped]); push!(cols, col); push!(vals, 0.5)
                end
            end
        end
    end
    return sparse(rows, cols, vals, dim, dim)
end

function extract_sparse_data(A::SparseMatrixCSC)
    dim = size(A, 1)
    
    # 1. Extract the diagonal. We convert it to a dense Vector.
    diags = Vector(diag(A))
    
    # Pre-allocate vectors for efficiency
    offdiags_flatten = Int[]
    sizehint!(offdiags_flatten, nnz(A) - dim) # Good starting size
    start_indices = zeros(Int, dim)

    # Julia's SparseMatrixCSC is in Column-Major format.
    # We iterate through each column `j`.
    for j in 1:dim
        # Record the starting position for this column's data.
        # This is the current length of the flattened array + 1.
        start_indices[j] = length(offdiags_flatten) + 1
        
        # A.colptr tells us the range of indices in A.rowval for column `j`.
        for ptr in A.colptr[j] : (A.colptr[j+1] - 1)
            # A.rowval gives us the row index `i` for the current non-zero element.
            i = A.rowval[ptr]
            
            # We only care about OFF-diagonal elements.
            if i != j
                push!(offdiags_flatten, i)
            end
        end
    end
    
    val = A[1, offdiags_flatten[1]]  # Just a placeholder, assuming uniform off-diagonal values

    return diags, offdiags_flatten, start_indices, val
end

function get_diags_and_offdiagonals_by_value(mat::AbstractMatrix{T}; tol=1e-14, round_digits=15) where {T<:Real}
    dim = size(mat, 1)

    # Map: val => list of connections for each row
    value_to_conn = Dict{Float64, Vector{Vector{Int}}}()

    # Collect diagonal entries
    diagonals = Vector{Float64}(undef, dim)

    for i in 1:dim
        diagonals[i] = round(mat[i, i]; digits=round_digits)  # store diagonal value

        row_conns = Dict{Float64, Vector{Int}}()
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            if i == j  # skip diagonal
                continue
            end
            v = mat[i, j]
            if abs(v) < tol
                continue
            end
            v_rounded = round(v; digits=round_digits)
            push!(get!(row_conns, v_rounded, Int[]), j)
        end
        for (v, js) in row_conns
            if !haskey(value_to_conn, v)
                value_to_conn[v] = [Int[] for _ in 1:dim]
            end
            value_to_conn[v][i] = js
        end
    end

    # Now flatten each list
    offdiagonals = Tuple{Float64, Vector{Int}, Vector{Int}}[]
    for (v, conn_lists) in sort(collect(value_to_conn), by=first)
        flat = Int[]
        starts = Int[]
        idx = 1
        for js in conn_lists
            push!(starts, idx)
            append!(flat, js)
            idx += length(js)
        end
        push!(offdiagonals, (v, flat, starts))
    end

    # length(offdiagonals) == 1 && (return diagonals, offdiagonals[1])  # If only one off-diagonal value, return it directly
    return diagonals, offdiagonals
end







# function mapvec_vega!(
#     diags::AbstractVector{Float64},
#     offdiagonals::AbstractVector,
# )
#     return @inline (Y,X) -> begin
        
#         # @. Y = diags * X
        
#         for (val, flat, start_inds) in offdiagonals
#             # total_sum_val = 0.
#             for i in eachindex(diags)
#                 @inbounds start = start_inds[i]
#                 @inbounds stop = i == length(start_inds) ? length(flat) : start_inds[i + 1] - 1

#                 sum_val = 0.0
#                 for j in start:stop
#                     @inbounds sum_val += X[flat[j]]
#                 end

#                 # @inbounds Y[i] += sum_val * val

#                 @inbounds Y[i] = diags[i] * X[i] + sum_val * val
#                 # @inbounds Y[i] = muladd(diags[i], X[i], val * sum_val)
#             end
#         end
#     end
# end


function mapvec_polfed!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    val::Float64
)
    return @inline (Y,X) -> begin
        for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], val * sum_val)
        end
    end
end






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


function make_loop(use_threads::Bool)
    return (range, body) -> begin
        if use_threads
            Threads.@threads for i in range
                body(i)
            end
        else
            for i in range
                body(i)
            end
        end
    end
end

# function make_loop(use_threads::Bool)
#     if use_threads
#         return function(range, f)
#             Threads.@threads for i in range
#                 f(i)
#             end
#         end
#     else
#         return function(range, f)
#             for i in range
#                 f(i)
#             end
#         end
#     end
# end



function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Tuple{<:Real, Vector{Int}, Vector{Int}},
    loop::Function,   # the loop is injected
)
    (val, offdiags_flatten, start_indices) = offdiagonals

    # loop(eachindex(X)) do i    
    loop(eachindex(start_indices), i -> begin
        Y_i = mapping_state_i(X, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds Y[i] = Y_i
    end)
end



function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Real, Vector{Int}, Vector{Int}}},
    loop::Function,
)
    @. Y = diagonals * X
    
    for (val, offdiags_flatten, start_indices) in offdiagonals
        # loop(eachindex(X)) do i
        loop(eachindex(start_indices), i -> begin
            Y_off_val_i = mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)
            @inbounds Y[i] += Y_off_val_i
        end)
    end
end






function mapping2!(
    Y::AbstractVector{T},
    X::AbstractVector{T},
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Real, Vector{Int}, Vector{Int}}},
    loop::Function,
) where {T<:Real}
    # loop(eachindex(X)) do i
    loop(eachindex(X), i -> begin
        Y_off_i = T(0.0)
        for (val, offdiags_flatten, start_indices) in offdiagonals
            Y_off_i += mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)
        end

        @inbounds Y[i] = diagonals[i]*X[i] + Y_off_i
    end)
end




println("Compare the two functions for extracting the sparse data: ")

L= parse(Int, ARGS[1])
mat = construct_xxz_spin_sector(L, 1.234, L÷2)
diags1, off_flat1, start_inds1, val1   = @time extract_sparse_data(mat)
diags2, offdiagonals2 = @time get_diags_and_offdiagonals_by_value(mat)
(val2, off_flat2, start_inds2) = offdiagonals2[1]  # Assuming uniform off-diagonal values for comparison



println("Are the diagonals equal? ", all(diags1 .≈ diags2))
println("Are the offdiagonal flattened arrays equal? ", all(off_flat1 .== off_flat2))
println("Are the start indices equal? ", all(start_inds1 .== start_inds2))
println("Are the offdiagonal values equal? ", val1 ≈ val2)


loop = make_loop(false)

# map_vega = mapvec_vega!(diags2, offdiagonals2)  
map_polfed = mapvec_polfed!(diags2, off_flat2, start_inds2, val2)
map_new1_vec =(Y,X) -> mapping!(Y,X,diags2,offdiagonals2, loop)
map_new1_tup =(Y,X) -> mapping!(Y,X,diags2,offdiagonals2[1], loop)
map_new2_vec =(Y,X) -> mapping2!(Y,X,diags2,offdiagonals2, loop)
mulmul = (Y,X) -> mul!(Y,mat,X)


x = rand(length(diags2))
y = similar(x)
y1 = similar(x)
y2 = similar(x)
y3 = similar(x)
y4 = similar(x)


println("Compare mappings: ")
mulmul(y, x)
map_polfed(y1, x)
map_new1_vec(y2, x)
map_new1_tup(y3, x)
map_new2_vec(y4, x)
println("Are the results equal? ", all(y .≈ y1))
println("Are the results equal? ", all(y .≈ y2))
println("Are the results equal? ", all(y .≈ y3))
println("Are the results equal? ", all(y .≈ y4))


println("Benchmarking the two mapping functions: ")
@btime $mulmul($y, $x)
@btime $map_polfed($y1, $x)
@btime $map_new1_vec($y2, $x)
@btime $map_new1_tup($y3, $x)
@btime $map_new2_vec($y4, $x)



#     diags::Vector{Float64},
#     offdiags_flatten::Vector{Int},
#     start_indices::Vector{Int},
#     J::Float64
# )
#     return (Y,X) -> begin
#         J_half =J/2
#         for i in eachindex(start_indices)
#             start = start_indices[i]
#             @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
#             sum_val = 0.0
#             for j in start:stop
#                 @inbounds sum_val += X[offdiags_flatten[j]]
#             end
#             @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
#         end
#     end
# end


# function mapvec_with_xxz_parallel!(
#     diags::Vector{Float64},
#     offdiags_flatten::Vector{Int},
#     start_indices::Vector{Int},
#     J::Float64
# )
#     return (Y,X) -> begin
#         J_half =J/2
#         @threads for i in eachindex(start_indices)
#             start = start_indices[i]
#             @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
#             sum_val = 0.0
#             for j in start:stop
#                 @inbounds sum_val += X[offdiags_flatten[j]]
#             end
#             @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
#         end
#     end
# end





# display(mat)

# display(diags2)
# display(off_flat2)
# display(start_inds2)
# display(val2)

