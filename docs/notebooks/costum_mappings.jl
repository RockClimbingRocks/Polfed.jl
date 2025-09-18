using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Base.Threads

# ==============================================================================
# 1. SETUP: Functions to generate the test data
# ==============================================================================

"""
Constructs the Hamiltonian for the XXZ model in a specific magnetization sector.
"""
function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
    basis = [b for b in 0:2^L-1 if count_ones(b) == Nup]
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))
    rows, cols, vals = Int[], Int[], Float64[]
    for (col, state) in enumerate(basis)
        diag_val = 0.0
        for i in 1:L
            j = i % L + 1
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1
            diag_val += delta * (0.5 - si) * (0.5 - sj)
            if si != sj
                flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                if haskey(bmap, flipped)
                    push!(rows, bmap[flipped]); push!(cols, col); push!(vals, 0.5)
                end
            end
        end
        push!(rows, col); push!(cols, col); push!(vals, diag_val)
    end
    return sparse(rows, cols, vals, dim, dim)
end

"""
Extracts data from a SparseMatrixCSC into the format required by the custom mappers.
"""
function extract_sparse_data(A::SparseMatrixCSC)
    dim = size(A, 1)
    
    # IMPORTANT FIX: diag(A) on a sparse matrix returns a SparseVector.
    # We must convert it to a dense Vector for our functions.
    diags = Vector(diag(A))
    
    offdiags_flatten = Int[]
    sizehint!(offdiags_flatten, nnz(A) - dim)
    start_indices = zeros(Int, dim)

    for j in 1:dim # Iterate through columns
        start_indices[j] = length(offdiags_flatten) + 1
        for ptr in A.colptr[j]:(A.colptr[j+1] - 1)
            i = A.rowval[ptr]
            if i != j # If it's an off-diagonal element
                push!(offdiags_flatten, i)
            end
        end
    end
    return diags, offdiags_flatten, start_indices
end


# ==============================================================================
# 2. IMPLEMENTATION: The three functions to be benchmarked
# ==============================================================================

"""
A serial (single-threaded) custom matrix-vector multiplication function.
"""
function costum_map_serial!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2
    len = length(start_indices)

    return (Y, X) -> begin
        for row in 1:len
            @inbounds start = start_indices[row]
            @inbounds stop  = (row == len) ? length(offdiags_flatten) : start_indices[row+1]-1
            
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            
            @inbounds Y[row] = muladd(diags[row], X[row], J_half * sum_val)
        end
        return Y
    end
end

"""
A parallel (multi-threaded) custom matrix-vector multiplication function.
"""
function costum_map_threads!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2
    len = length(start_indices)

    return (Y, X) -> begin
        # @threads automatically divides the loop into contiguous blocks,
        # one for each available thread.
        @threads for row in 1:len
            @inbounds start = start_indices[row]
            @inbounds stop  = (row == len) ? length(offdiags_flatten) : start_indices[row+1]-1
            
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            
            @inbounds Y[row] = muladd(diags[row], X[row], J_half * sum_val)
        end
        return Y
    end
end


"""
A parallel (multi-threaded) custom matrix-vector multiplication function.
"""
function costum_map_threads!2(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2
    len = length(start_indices)

    return (Y, X) -> begin
        # @threads automatically divides the loop into contiguous blocks,
        # one for each available thread.
        @threads for t in 1:2
            t_start = t==1 ? 1 : len÷2+1
            t_end = t==1 ? len÷2 : len

            for row in t_start:t_end
                @inbounds start = start_indices[row]
                @inbounds stop  = (row == len) ? length(offdiags_flatten) : start_indices[row+1]-1
                
                sum_val = 0.0
                for j in start:stop
                    @inbounds sum_val += X[offdiags_flatten[j]]
                end
                
                @inbounds Y[row] = muladd(diags[row], X[row], J_half * sum_val)
            end
        end
        return Y
    end
end


# ==============================================================================
# 3. BENCHMARKING SCRIPT
# ==============================================================================

# --- Setup Parameters ---
L = 18
Nup = L ÷ 2
delta = 1.0

println("="^60)
println("Benchmarking Sparse Matrix-Vector Multiplication")
println("Parameters: L=$L, Nup=$Nup, delta=$delta")
println("Julia threads available: $(Threads.nthreads())")
println("="^60)

# --- Generate Data ---
print("Setting up data...")
A = construct_xxz_spin_sector(L, delta, Nup)
dim = size(A, 1)
diags, offdiags, starts = extract_sparse_data(A)
J = 1.0 # The custom mapper uses J, which is 2 * the off-diagonal value in A

X = rand(Float64, dim) # Input vector
Y = zeros(Float64, dim) # Output vector (pre-allocated)
println(" Done. (Matrix dimension: $dim x $dim)")

# --- Create the mapping functions ---
map_serial!  = costum_map_serial!(diags, offdiags, starts, J)
map_threads! = costum_map_threads!(diags, offdiags, starts, J)
map_threads!2 = costum_map_threads!2(diags, offdiags, starts, J)

# --- Verification Step (Important!) ---
# Ensure all methods produce the same result before trusting the benchmarks
Y_blas = A * X
Y_serial = map_serial!(similar(Y), X)
@assert Y_blas ≈ Y_serial "ERROR: BLAS and Serial results do not match!"
println("\nVerification successful: All methods produce the same output.")

# --- Run Benchmarks ---
println("\nRunning benchmarks...\n")

println("1. LinearAlgebra.mul! (Optimized BLAS):")
@btime mul!($Y, $A, $X)

println("\n2. Custom Serial Mapper (Single-Threaded):")
@btime $map_serial!($Y, $X)

println("\n3. Custom Parallel Mapper (Multi-Threaded):")
@btime $map_threads!($Y, $X)

println("\n4. Custom Parallel Mapper (Multi-Threaded, 2 Threads):")
@btime $map_threads!2($Y, $X)

println("\n" * "="^60)
println("Benchmark complete.")