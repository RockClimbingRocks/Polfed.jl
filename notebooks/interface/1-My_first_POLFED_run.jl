
using SparseArrays, LinearAlgebra
include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
using .Polfed


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



println("First run:")
begin

    L = 14; Nup = L÷2; delta=1.0
    mat = construct_xxz_spin_sector(L, delta, Nup)
    target = 0.0; howmany = 1000
    v0 = rand(size(mat,1)); v0 ./= norm(v0)



    LinearAlgebra.BLAS.set_num_threads(1)
    _, _ = polfed(mat, v0, howmany, target) # Warmup run 
    vals, vecs = @time polfed(mat, v0, howmany, target) 
end



println();println();println();
println("Second that shows report:")
begin

    LinearAlgebra.BLAS.set_num_threads(1)
    vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true)
    Polfed.display_report(report)
end




println();println();println();
println("Second run with block size 4 and 4 threads with parallelilzation over columns:")
begin
    LinearAlgebra.BLAS.set_num_threads(4)
        # LinearAlgebra.BLAS.set_num_threads(4)

    v0_ = rand(size(mat,1), 4); v0 = Matrix(qr(v0_).Q)

    vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true)
    Polfed.display_report(report)
end





