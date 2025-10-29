
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




# function clenshaw_reocurence_relation_vec_kernel_spin_onehalf(b1::CuDeviceVector, b2::CuDeviceVector, b3::CuDeviceVector, c::Real, X::CuDeviceVector, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

#     @inbounds @fastmath if i <= basis_length 
#         d = diags[i]
#         x = b2[i]

#         offdiag_val = 0.0 
#         @simd for j in 0:Leff
#             newstate = (i-1) ⊻ (1 << j) 
#             row = newstate + 1 
#             offdiag_val += b2[row]
#         end     
#         yi = d*x + offdiag_val*hx

#         b1[i] = c*X[i] + 2*yi - b3[i]
#     end

#     nothing
# end


# function clenshaw_finalsum_vec_kernel_spin_onehalf(b1::CuDeviceVector, b2::CuDeviceVector, c::Real, Y::CuDeviceVector, X::CuDeviceVector, hx::Float64, diags::CuDeviceVector, Leff::Int64, basis_length::Int64)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

#     @inbounds @fastmath if i <= basis_length 
#         d = diags[i]
#         x = b1[i]
#         # X here goes to b1

#         offdiag_val = 0.0 
#         @simd for j in 0:Leff
#             newstate = (i-1) ⊻ (1 << j) 
#             row = newstate + 1 
#             offdiag_val += b1[row] 
#         end     
#         y = d*x + offdiag_val*hx

#         Y[i] = c*X[i] + y - b2[i]
#     end

#     nothing
# end





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
nt = Base.Threads.nthreads()
ncols = nt ÷ nt_per_col
J=1.

println("L: ", L)
println("howmany: ", howmany)
println("Number of threads: ", nt)
println("Number of columns (workers): ", ncols)
println("Number of threads per column: ", nt_per_col)
println("Benchmark polfed WITH CUSTOM CLENSHAW function (L=$(L)): ")


mat = construct_xxz_spin_sector(L, 1.234, L÷2)
diags, off_flat, start_inds = extract_sparse_data(mat)
vmap = mapvec_with_xxz_parallel!(diags, off_flat, start_inds, J)


v02 = rand(size(mat,1)); v02 = v02 / norm(v02)
Emin = first(collect(Polfed.Lanczos.lanczos(vmap, v02, 1; which=:SR, maxdim=1000)[1]))
Emax = last(collect(Polfed.Lanczos.lanczos(vmap, v02, 1; which=:LR,  maxdim=1000)[1]))

v0_ = rand(size(mat,1), ncols); v0 = Matrix(qr(v0_).Q)

a, b = (Emax-Emin)/2, (Emax+Emin)/2
d_r = @. (diags-b)/a
J_r =  J/a

vmap_r! = mapvec_with_xxz_parallel!(d_r,off_flat, start_inds,J_r)
crr, cfs = clenshaw_with_xxz_parallel!(d_r, off_flat, start_inds, J_r)


st = Polfed.SpectralTransformConfig(;
    parallelization=Polfed.TwoLevelParallel(nt_per_col),s
)

times = zeros(Float64, N)
for i in 1:N
    time = @elapsed begin vals, vecs, report = @time Polfed.polfed(vmap, v0, howmany, 0.; optimize_mapping=true, produce_report=true, spectral_transform=st) end
    Polfed.display_report(report)

    times[i] = time
    println("Iteration $i / $N took $time seconds.")
    println()
    println()
    println()
    println()
end


h5file = h5open("bpc_L$(L)_N$(howmany)_nt$(nt)_ntpc$(nt_per_col)_avg$(N).h5", "w") do file
    file["L"] = L
    file["howmany"] = howmany
    file["nt"] = nt
    file["nt_per_col"] = nt_per_col
    file["times"] = times
    file["time_avg"] = mean(times)
    file["time_std"] = std(times)
    file["time_typ"] = exp(mean(log.(times)))
end


println("Results saved to HDF5 file.")


