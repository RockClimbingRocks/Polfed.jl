


"""
    are_vals_in_true(vals, vals_true; atol=1e-8)

Check whether the sequence `vals` appears (within tolerance `atol`) 
as a consecutive subsequence of `vals_true`.

Returns `true` if all values match within tolerance, otherwise `false`.
"""
function are_vals_in_true(vals::AbstractVector, vals_true::AbstractVector; atol=1e-8)
    # Find the closest index in `vals_true` to vals[1]
    i_min = findmin(abs.(vals_true .- vals[1]))[2]
    i_max = i_min + length(vals) - 1

    # Ensure the slice fits inside vals_true
    if i_max > length(vals_true)
        return false
    end

    display(abs.(vals .- view(vals_true, i_min:i_max)))

    # Check all values with broadcasting
    return all(abs.(vals .- view(vals_true, i_min:i_max)) .< atol)
end





using SparseArrays, LinearAlgebra, CUDA
# using QSystem

include("../src/Polfed.jl")
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



function test_polfed() 

    L =18
    mat = construct_xxz_spin_sector(L, 0.123, Int(L÷2)) # XXZ model with delta=0.0

    # display(mat)
    D = size(mat, 1)

    # x0_ = rand(D)
    # x0 = x0_ ./ norm(x0_)
    # x0_ = rand(D,4)
    # x0 = Matrix(qr(x0_).Q) # orthonormalize
    # mapping = Polfed.MappingConfig(
    #     parallel_strategy=Polfed.MulColsParallel()
    # )

    x0_ = rand(D,4)
    x0 = Matrix(qr(x0_).Q) # orthonormalize
    mapping = Polfed.MappingConfig(
        parallel_strategy=Polfed.TwoLevelParallel(1)
    )


    howmany = 500
    vals, vecs, report = Polfed.polfed(mat, x0, howmany, 0.; produce_report=true, mapping=mapping)
    Polfed.display_report(report)

    # mat_dense = Matrix(mat)
    # vals_true = eigvals!(mat_dense)

    # r = are_vals_in_true(vals, vals_true)
    # println("Are all values in true? ", r)



    # vals, vecs = Lanczos.lanczos(mat, x0, howmany; maxdim=1000, tol=1e-14, eigentol=1e-10)
    # println("Computed eigenvalues:\n", vals)
    # println("Corresponding eigenvectors:\n")
    # println(vecs)
end


function test_lanczos_CUDA() 
    L =22
    mat = construct_xxz_spin_sector(L, 0.123, Int(L÷2)) 
    mat_cu = CUDA.CUSPARSE.CuSparseMatrixCSR(mat) # convert to CuMatrix for GPU
    f!(Y,X) = mul!(Y, mat_cu, X)
    D = size(mat, 1)

    x0_ = CUDA.rand(Float64, D,4)
    x0 = CuMatrix(qr(x0_).Q) # orthonormalize
    howmany = 500


    vals, vecs, report = Polfed.polfed(f!, x0, howmany, 0.; produce_report=true)
    # Polfed.lanczos(f!, x0, howmany; maxdim=1000, tol=1e-14, eigentol=1e-8, basistype=Lanczos.HybridMatrixBasis)
    Polfed.display_report(report)



    # vals_true, vecs_true = Lanczos2.lanczosmethod(f!, x0, howmany; maxdim = 1000, tol = 1e-14, eigentol = 1e-8)

    # println(Vector(vals_true) ≈ Vector(vals))
    # println(Matrix(vecs_true) ≈ Matrix(vecs))

    # errs = abs.(Matrix(vecs_true) - Matrix(vecs))


    # # println(errs)
    # for col in eachcol(errs)
    #     println("Error norm for column: ", norm(col))
    #     println("Max error: ", maximum(col))
    # end
    # println("Max error (norm): ", maximum(errs))
    # println("Max error (norm): ", norm(errs))
end


test_polfed()
