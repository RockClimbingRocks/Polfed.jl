
println("Activated environment: ", Base.active_project())
include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
using .Polfed

using SparseArrays, LinearAlgebra, CUDA
using  Base.Threads

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



function run_benchmark()
    L = 22; Nup = L÷2; delta=1.0
    mat = construct_xxz_spin_sector(L, delta, Nup)
    target = 0.0; howmany = parse(Int64,ARGS[1])
    

    nt = Base.Threads.nthreads()
    nt_per_col = 2
    ncols = nt ÷ nt_per_col
    v0_ = rand(size(mat,1), ncols); v0 = Matrix(qr(v0_).Q)
    spec_trans = Polfed.SpectralTransformConfig(;parallelization=Polfed.TwoLevelParallel(nt_per_col))
    

    _, _, report = @time polfed(mat, v0, howmany, target; 
        produce_report=true,
        spectral_transform=spec_trans
    )



    h5file = h5open("benchmark_time_L=$(L)_N=$(howmany)_nt=$(nt).h5", "w")
    h5file["time"] = time
    h5file["L"] = L
    h5file["nt"] = nt
    close(h5file)


    
    Polfed.display_report(report)
end


run_benchmark()