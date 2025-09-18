
println("Activated environment: ", Base.active_project())
include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
using .Polfed

using SparseArrays, LinearAlgebra, CUDA, HDF5
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
    
    return diags, offdiags_flatten, start_indices
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
    L = parse(Int64,ARGS[1])
    howmany = parse(Int64,ARGS[2])
    paralization_type = ARGS[3]
    nt_per_col = parse(Int64,ARGS[4])
    norm_to = parse(Float64, ARGS[5])

    println("L = $L")
    println("howmany = $howmany")
    println("paralization_type = $paralization_type")
    println("nt_per_col = $nt_per_col")
    flush(stdout)

    Nup = L÷2; delta=1.0; J=1.0
    mat = construct_xxz_spin_sector(L, delta, Nup)
    diags, offdiags_flatten, start_indices = extract_sparse_data(mat)
    target = 0.0


    nt = Base.Threads.nthreads()
    spec_trans = nothing
    v0 = nothing
    vmap = nothing
    vmap_rescaled = nothing

    if paralization_type == "1"
        v0_ = rand(size(mat,1), nt); v0 = Matrix(qr(v0_).Q)
        spec_trans = Polfed.SpectralTransformConfig(;parallelization=Polfed.MulColsParallel())
        vmap = costum_map_serial!(diags, offdiags_flatten, start_indices, J)

        v0 = pu.Vector(x0[:,1])        
        @time Emin = first(collect(Polfed.Lanczos.lanczos(vmap, v0, 1; which=:smallest, maxdim=1000)[1]))
        @time Emax = last(collect(Polfed.Lanczos.lanczos(vmap, v0, 1; which=:largest,  maxdim=1000)[1]))
        a = (Emax-Emin)/2
        b = (Emax+Emin)/2
        diags_rescaled =  @. diags / a - (b/a)
        J_rescaled = J / a

        vmap_rescaled = costum_map_threads!(diags_rescaled, offdiags_flatten, start_indices, J_rescaled)
    elseif paralization_type == "2"
        ncols = nt ÷ nt_per_col
        v0_ = rand(size(mat,1), ncols); v0 = Matrix(qr(v0_).Q)
        vmap = costum_map_threads!(diags, offdiags_flatten, start_indices, J)
        
        v02 = Vector(v0[:,1])        
        @time Emin = first(collect(Polfed.Lanczos.lanczos(vmap, v02, 1; which=:smallest, maxdim=1000)[1]))
        @time Emax = last(collect(Polfed.Lanczos.lanczos(vmap, v02, 1; which=:largest,  maxdim=1000)[1]))
        a = (Emax-Emin)/2
        b = (Emax+Emin)/2
        diags_rescaled =  @. diags / a - (b/a)
        J_rescaled = J / a
        vmap_rescaled = costum_map_threads!(diags_rescaled, offdiags_flatten, start_indices, J_rescaled)
        
        spec_trans = Polfed.SpectralTransformConfig(;
            parallelization=Polfed.TwoLevelParallel(nt_per_col),
            f!_rescaled = vmap_rescaled
        )
    else 
        throw(ArgumentError("Unknown paralization_type: $paralization_type"))
    end
    
    
    spec_trans.normalization = norm_to
    spec_trans.cutoff = 0.17*spec_trans.normalization


    time = @elapsed _, _, report = @time polfed(vmap, v0, howmany, target;
        produce_report=true,
        spectral_transform=spec_trans,
    )

    flush(stdout)
    h5file = h5open("benchmark_norm=$(norm_to)_rescaled_parallel=$(paralization_type)_L=$(L)_N=$(howmany)_nt=$(nt)_ntpc=$(nt_per_col).h5", "w")
    h5file["time"] = time
    h5file["norm_to"] = norm_to
    h5file["L"] = L
    h5file["howmany"] = howmany
    h5file["nt"] = nt
    h5file["nt_per_col"] = nt_per_col
    close(h5file)

    Polfed.display_report(report)

    flush(stdout)
end


run_benchmark()