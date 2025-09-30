
using SparseArrays, LinearAlgebra, BenchmarkTools, Base.Threads
using HDF5
include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
using .Polfed


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
    
    return diags, offdiags_flatten, start_indices
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


L = parse(Int, ARGS[1])
howmany = parse(Int, ARGS[2])
nt_per_col = parse(Int, ARGS[3])
N = parse(Int, ARGS[4])
benc_type = ARGS[5]  # "opt", "map", or "clenshaw"

nt = Base.Threads.nthreads()
ncols = nt ÷ nt_per_col
J=1.



println("L: ", L)
println("howmany: ", howmany)
println("benc_type: ", benc_type)
println("N (number of runs for averaging): ", N)
println("Number of threads: ", nt)
println("Number of columns (workers): ", ncols)
println("Number of threads per column: ", nt_per_col)


mat = construct_xxz_spin_sector(L, 1.234, L÷2)
diags, off_flat, start_inds = extract_sparse_data(mat)
vmap! = mapvec_with_xxz!(diags, off_flat, start_inds, J)
vmap_parallel! = mapvec_with_xxz_parallel!(diags, off_flat, start_inds, J)


v02 = rand(size(mat,1)); v02 = v02 / norm(v02)
Emin = first(collect(Polfed.Lanczos.lanczos(vmap!, v02, 1; which=:smallest, maxdim=1000)[1]))
Emax = last(collect(Polfed.Lanczos.lanczos(vmap!, v02, 1; which=:largest,  maxdim=1000)[1]))

v0_ = rand(size(mat,1), ncols); v0 = Matrix(qr(v0_).Q)

a, b = (Emax-Emin)/2, (Emax+Emin)/2
d_r = @. (diags-b)/a
J_r =  J/a

vmap_r! = mapvec_with_xxz!(d_r,off_flat, start_inds,J_r)
vmap_r_parallel! = mapvec_with_xxz_parallel!(d_r,off_flat, start_inds,J_r)
crr_r!, cfs_r! = clenshaw_with_xxz!(d_r, off_flat, start_inds, J_r)
crr_r_parallel!, cfs_r_parallel! = clenshaw_with_xxz_parallel!(d_r, off_flat, start_inds, J_r)



input_mat_or_vec = nothing
key_word_args = Dict{Symbol,Any}()

if benc_type == "type1"
    input_mat_or_vec = mat

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.MulColsParallel(),
    )

    key_word_args[:spectral_transform] = st
    key_word_args[:produce_report] = true


elseif benc_type == "type2"
    input_mat_or_vec = vmap!

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.MulColsParallel(),
    )

    key_word_args[:spectral_transform] = st
    key_word_args[:produce_report] = true

elseif benc_type == "type3"
    input_mat_or_vec = vmap_parallel!

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.TwoLevelParallel(nt_per_col),
    )
    
    key_word_args[:spectral_transform] = st
    key_word_args[:produce_report] = true

elseif benc_type == "type4"
    input_mat_or_vec = vmap_parallel!

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.TwoLevelParallel(nt_per_col),
        f!_rescaled=vmap_r_parallel!,
    )
    
    key_word_args[:spectral_transform] = st
    key_word_args[:produce_report] = true

elseif benc_type == "type5"
    input_mat_or_vec = vmap_parallel!

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.TwoLevelParallel(nt_per_col),
        f!_rescaled=vmap_r_parallel!,
        clenshaw_recurrence=crr_r_parallel!,
        clenshaw_finalsum=cfs_r_parallel!,
    )
    
    key_word_args[:spectral_transform] = st
    key_word_args[:produce_report] = true

elseif benc_type == "type6"
    input_mat_or_vec = vmap!

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.MulColsParallel(),
        f!_rescaled=vmap_r!,
        clenshaw_recurrence=crr_r!,
        clenshaw_finalsum=cfs_r!,
    )
    
    key_word_args[:spectral_transform] = st
    key_word_args[:produce_report] = true

elseif benc_type == "type7"
    input_mat_or_vec = mat

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.MulColsParallel(),
    )
    
    key_word_args[:spectral_transform] = st
    key_word_args[:optimize_mapping] = true
    key_word_args[:produce_report] = true

elseif benc_type == "type8"
    input_mat_or_vec = mat

    st = Polfed.SpectralTransformConfig(;
        parallelization=Polfed.TwoLevelParallel(nt_per_col),
    )
    
    key_word_args[:spectral_transform] = st
    key_word_args[:optimize_mapping] = true
    key_word_args[:produce_report] = true
else
    error("Unknown benc_type: $benc_type")
end


times = zeros(Float64, N)
for i in 1:N
    time = @elapsed begin vals, vecs, report = @time Polfed.polfed(input_mat_or_vec, v0, howmany, 0.; key_word_args...) end
    Polfed.display_report(report)

    times[i] = time
    println("Iteration $i / $N took $time seconds.")
    println()
    println()
    println()
    println()
end


h5file = h5open("$(benc_type)_L$(L)_N$(howmany)_nt$(nt)_ntpc$(nt_per_col)_avg$(N).h5", "w") do file
    file["L"] = L
    file["howmany"] = howmany
    file["N"] = N
    file["benc_type"] = benc_type
    file["nt"] = nt
    file["nt_per_col"] = nt_per_col
    file["times"] = times
    file["time_avg"] = mean(times)
    file["time_std"] = std(times)
    file["time_typ"] = exp(mean(log.(times)))
end


println("Results saved to HDF5 file.")


