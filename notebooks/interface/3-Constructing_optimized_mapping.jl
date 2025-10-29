
using SparseArrays, LinearAlgebra, BenchmarkTools, Base.Threads
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



L = parse(Int, ARGS[1])
howmany = parse(Int, ARGS[2])
nt_per_col =  parse(Int, ARGS[3])   


println("Benchmark the mapping (L=$(L)): ")
begin
    J=1.
    mat = construct_xxz_spin_sector(L, 1.234, L÷2)
    diags, offdiags_flatten, start_indices = extract_sparse_data(mat)

    X = rand(size(mat,1)); Y = similar(X)

    mapv! = mapvec_with_xxz!(diags, offdiags_flatten, start_indices, J)
    LinearAlgebra.BLAS.set_num_threads(1)
    @btime mul!($Y, $mat, $X) 
    @btime $mapv!($Y, $X)
end




# println("Benchmark regular POLFED with custom mapping with MulColsParallel (for L=$(L)): ")
# begin
#     J=1.
#     mat = construct_xxz_spin_sector(L, 1.234, L÷2)
#     diags, off_flat, start_inds = extract_sparse_data(mat)
    
#     nt = Base.Threads.nthreads()


#     vmap = mapvec_with_xxz!(diags, off_flat, start_inds, J) 
#     v0_ = rand(size(mat,1), nt); v0 = Matrix(qr(v0_).Q)
#     st = Polfed.SpectralTransformConfig(;parallelization=Polfed.MulColsParallel())
#     vals, vecs, report= @time Polfed.polfed(vmap, v0, howmany, 0.; spectral_transform=st, produce_report=true)
#     Polfed.display_report(report)
# end





# println("Benchmark regular POLFED with custom mapping with TwoLevelParallel($(nt_per_col)) (for L=$(L)): ")
# begin
#     J=1.
#     mat = construct_xxz_spin_sector(L, 1.234, L÷2)
#     diags, off_flat, start_inds = extract_sparse_data(mat)
    
#     nt = Base.Threads.nthreads()
#     ncols = nt ÷ nt_per_col

#     vmap = mapvec_with_xxz_parallel!(diags, off_flat, start_inds, J) 
#     v0_ = rand(size(mat,1), ncols); v0 = Matrix(qr(v0_).Q)
#     st = Polfed.SpectralTransformConfig(;parallelization=Polfed.TwoLevelParallel(2))
#     vals, vecs, report= @time Polfed.polfed(vmap, v0, howmany, 0.; spectral_transform=st, produce_report=true)
#     Polfed.display_report(report)
# end





println("Benchmark POLFED with custom mapping with TwoLevelParallel($(nt_per_col)) with rescaled function (for L=$(L)): ")
begin
    v0 = rand(size(mat,1)); v0 = v0 / norm(v0)
    Emin = first(collect(Polfed.Lanczos.lanczos(vmap, v0, 1; which=:SR, maxdim=1000)[1]))
    Emax = last(collect(Polfed.Lanczos.lanczos(vmap, v0, 1; which=:LR,  maxdim=1000)[1]))

    a, b = (Emax-Emin)/2, (Emax+Emin)/2
    d_r = @. (diags-b)/a
    J_r =  J/a

    vmap = mapvec_with_xxz_parallel!(diags, off_flat, start_inds, J) 
    vmap_r! = mapvec_with_xxz_parallel!(d_r, off_flat, start_inds, J_r)
    
    nt = Base.Threads.nthreads()
    ncols = nt ÷ nt_per_col
    v0_ = rand(size(mat,1), ncols); v0 = Matrix(qr(v0_).Q)
    st = Polfed.SpectralTransformConfig(;parallelization=Polfed.TwoLevelParallel(nt_per_col), f!_rescaled=vmap_r!)
    vals, vecs = @time Polfed.polfed(vmap, v0, howmany, 0.; spectral_transform=st)
end