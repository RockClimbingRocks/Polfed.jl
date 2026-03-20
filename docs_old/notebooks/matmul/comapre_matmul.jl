
using LinearAlgebra, Base.Threads, SparseArrays
using BenchmarkTools


println("--- Julia Threading Info ---")
println("Threads.nthreads(): $(Threads.nthreads())")
println("Threads.threadid(): $(Threads.threadid())")
println("JULIA_NUM_THREADS env var: $(get(ENV, "JULIA_NUM_THREADS", "Not Set"))")

println("\n--- BLAS Threading Info ---")
println("BLAS.get_num_threads(): $(BLAS.get_num_threads())")
println("OMP_NUM_THREADS env var: $(get(ENV, "OMP_NUM_THREADS", "Not Set"))")
println("OPENBLAS_NUM_THREADS env var: $(get(ENV, "OPENBLAS_NUM_THREADS", "Not Set"))")
println("MKL_NUM_THREADS env var: $(get(ENV, "MKL_NUM_THREADS", "Not Set"))")

println("\n--- General Process Info ---")
println("Process ID (PID): $(getpid())")
println("Hostname: $(gethostname())")

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
            # --- S^z_i S^z_j diagonal term ---
            SzSz = (0.5 - si) * (0.5 - sj)  # spin-½: Sz = ±½
            push!(rows, col); push!(cols, col); push!(vals, delta * SzSz)
            # --- S⁺_i S⁻_j + h.c. (flip-flop term) ---
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


L= 20
delta=1.
Nup=L÷2


mat = construct_xxz_spin_sector(L, delta, Nup)
D = size(mat,1)
X = rand(D); 
Y = similar(X)

BLAS.set_num_threads(1) # avoid BLAS multithreading inside our threads
@btime mul!($Y, $mat, $X) # standard multiplication
@btime parallelize_mul_per_col!($Y, $mat, $X, 2)