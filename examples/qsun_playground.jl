using LinearAlgebra
using Printf
using Random
using SparseArrays

using Polfed.QSun: qsun_hamiltonian

function summarize_matrix(name::AbstractString, H)
    Hdense = Matrix(H)
    eigs = eigvals(Hermitian(Hdense))

    println("\n", name)
    println(repeat("-", length(name)))
    println("size       = ", size(H))
    println("nnz        = ", H isa SparseMatrixCSC ? nnz(H) : "dense")
    println("density    = ", @sprintf("%.3f", count(!iszero, Hdense) / length(Hdense)))
    println("Hermitian  = ", ishermitian(Hdense))
    println("spectrum   = [", @sprintf("%.6f", minimum(eigs)), ", ", @sprintf("%.6f", maximum(eigs)), "]")
    println("first eigs = ", eigs[1:min(5, end)])

    return eigs
end

function main()
    L_loc = 4
    L_grain = 2
    g0 = 1.0
    α = 0.55

    println("Quantum Sun playground")
    println("======================")
    println("Parameters: L_loc=$L_loc, L_grain=$L_grain, g0=$g0, α=$α")

    H_full = qsun_hamiltonian(
        L_loc,
        L_grain,
        g0,
        α;
        S=0.5,
        γ=1.0,
        w=0.5,
        hz=1.0,
        ζ=0.2,
        rng=MersenneTwister(1234),
        use_sparse=true,
    )
    summarize_matrix("Full Hilbert-space QSun", H_full)

    H_u1 = qsun_hamiltonian(
        L_loc,
        L_grain,
        g0,
        α;
        S=0.5,
        γ=1.0,
        w=0.5,
        hz=1.0,
        ζ=0.2,
        rng=MersenneTwister(1234),
        use_U1=true,
        S_z=0.0,
        use_sparse=true,
    )
    eigs_u1 = summarize_matrix("U(1)-conserving S_z=0 sector", H_u1)

    println("\nTry next")
    println("--------")
    println("1. Change L_loc, L_grain, α, or S_z above and rerun this file.")
    println("2. Pass H_full or H_u1 into Polfed.polfed for larger sparse systems.")
    println("3. Use exact eigvals only for tiny playground sizes like this one.")
    println("\nMiddle of the U(1) spectrum is around ", @sprintf("%.6f", eigs_u1[cld(length(eigs_u1), 2)]))

    return nothing
end

main()
