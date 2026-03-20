
using SparseArrays, LinearAlgebra, BenchmarkTools, Base.Threads
include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
# using .Polfed

using .Polfed: optimized_mapping!, optimized_clenshaw_recurrence_relation!, optimized_clenshaw_final_sum!, CPU, GPU, MulColsParallel, NoParallel, TwoLevelParallel


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




function mapvec_with_xxz!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    return (Y,X) -> begin
        J_half =J/2
        for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
        end
    end
end


function mapvec_with_xxz_parallel!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    return (Y,X) -> begin
        J_half =J/2
        @threads for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
        end
    end
end


function clenshaw_with_xxz!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2

    crr = @inline (b1::AbstractVector, b2::AbstractVector, b3::AbstractVector, c::Real, X::AbstractVector) -> begin

        for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b2[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b2[i], J_half * sum_val)
            @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
        end
    end


    cfs = @inline (b1::AbstractVector, b2::AbstractVector, c::Real, Y::AbstractVector, X::AbstractVector) -> begin

        for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b1[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b1[i], J_half * sum_val)
            @inbounds Y[i] = c*X[i] + yi - b2[i]
        end
    end


    return crr, cfs
end


function clenshaw_with_xxz_parallel!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2

    crr = @inline (b1::AbstractVector, b2::AbstractVector, b3::AbstractVector, c::Real, X::AbstractVector) -> begin

        @threads for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b2[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b2[i], J_half * sum_val)
            @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
        end
    end


    cfs = @inline (b1::AbstractVector, b2::AbstractVector, c::Real, Y::AbstractVector, X::AbstractVector) -> begin

        @threads for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b1[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b1[i], J_half * sum_val)
            @inbounds Y[i] = c*X[i] + yi - b2[i]
        end
    end


    return crr, cfs
end



println("Benchmarking custom vs Polfed functions:")

L = parse(Int, ARGS[1])
delta = 1.234
Nup = L ÷ 2
J = 1.0

mat = construct_xxz_spin_sector(L, delta, Nup)
diags, off_flat, start_inds, val = extract_sparse_data(mat)

# Custom functions
map_custom = mapvec_with_xxz!(diags, off_flat, start_inds, J)
crr_custom, cfs_custom = clenshaw_with_xxz!(diags, off_flat, start_inds, J)


offdiagonals = (val, off_flat, start_inds)
parallel_strategy = MulColsParallel()
# Polfed functions
map_polfed = optimized_mapping!(diags, offdiagonals, parallel_strategy)
crr_polfed = optimized_clenshaw_recurrence_relation!(diags, offdiagonals, parallel_strategy)
cfs_polfed = optimized_clenshaw_final_sum!(diags, offdiagonals, parallel_strategy)

x = rand(length(diags))
y1 = similar(x)
y2 = similar(x)
b1 = similar(x)
b2 = similar(x)
b3 = similar(x)

println("mapvec_with_xxz! vs optimized_mapping!")
@btime $map_custom($y1, $x)
@btime $map_polfed($y2, $x)
# println("Are results equal? ", all(y1 .≈ y2))

println("clenshaw_with_xxz! vs optimized_clenshaw_recurrence_relation!")
# Prepare random vectors for Clenshaw
b1 .= rand(length(diags))
b2 .= rand(length(diags))
b3 .= rand(length(diags))
c = 0.5

@btime $crr_custom($b1, $b2, $b3, $c, $x)
@btime $crr_polfed($b1, $b2, $b3, $c, 1, $x)

# Final sum
Y1 = similar(x)
Y2 = similar(x)
@btime $cfs_custom($b1, $b2, $c, $Y1, $x)
@btime $cfs_polfed($b1, $b2, $c, $Y2, $x)
# println("Are final Clenshaw results equal? ", all(Y1 .≈ Y2))



# println("Compare the two functions for extracting the sparse data: ")

# L= parse(Int, ARGS[1])
# mat = construct_xxz_spin_sector(L, 1.234, L÷2)
# diags1, off_flat1, start_inds1, val1   = @time extract_sparse_data(mat)
# diags2, offdiagonals2 = @time get_diags_and_offdiagonals_by_value(mat)
# (val2, off_flat2, start_inds2) = offdiagonals2[1]  # Assuming uniform off-diagonal values for comparison



# println("Are the diagonals equal? ", all(diags1 .≈ diags2))
# println("Are the offdiagonal flattened arrays equal? ", all(off_flat1 .== off_flat2))
# println("Are the start indices equal? ", all(start_inds1 .== start_inds2))
# println("Are the offdiagonal values equal? ", val1 ≈ val2)


# loop = make_loop(false)

# # map_vega = mapvec_vega!(diags2, offdiagonals2)  
# map_polfed = mapvec_polfed!(diags2, off_flat2, start_inds2, val2)
# map_new1_vec =(Y,X) -> mapping!(Y,X,diags2,offdiagonals2, loop)
# map_new1_tup =(Y,X) -> mapping!(Y,X,diags2,offdiagonals2[1], loop)
# map_new2_vec =(Y,X) -> mapping2!(Y,X,diags2,offdiagonals2, loop)
# mulmul = (Y,X) -> mul!(Y,mat,X)


# x = rand(length(diags2))
# y = similar(x)
# y1 = similar(x)
# y2 = similar(x)
# y3 = similar(x)
# y4 = similar(x)


# println("Compare mappings: ")
# mulmul(y, x)
# map_polfed(y1, x)
# map_new1_vec(y2, x)
# map_new1_tup(y3, x)
# map_new2_vec(y4, x)
# println("Are the results equal? ", all(y .≈ y1))
# println("Are the results equal? ", all(y .≈ y2))
# println("Are the results equal? ", all(y .≈ y3))
# println("Are the results equal? ", all(y .≈ y4))


# println("Benchmarking the two mapping functions: ")
# @btime $mulmul($y, $x)
# @btime $map_polfed($y1, $x)
# @btime $map_new1_vec($y2, $x)
# @btime $map_new1_tup($y3, $x)
# @btime $map_new2_vec($y4, $x)
