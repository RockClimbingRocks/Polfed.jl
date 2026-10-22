using Test
using Polfed
using LinearAlgebra
using SparseArrays
using Random

Q = Polfed.Models.QSun

@testset "quantum_sun (build_hamiltonian)" begin

    @testset "dimension, hermiticity, types" begin
        H = Q.build_hamiltonian(2, 2, 1.0, 0.5; rng = MersenneTwister(1))
        @test H isa SparseMatrixCSC
        @test size(H) == (16, 16)              # d^(L_loc+L_grain) = 2^4
        @test ishermitian(Matrix(H))

        Hd = Q.build_hamiltonian(2, 2, 1.0, 0.5; rng = MersenneTwister(1), use_sparse = false)
        @test Hd isa Matrix{Float64}
        @test Hd ≈ Matrix(H)                   # sparse and dense paths agree
    end

    @testset "reproducibility (seeded rng)" begin
        H1 = Q.build_hamiltonian(2, 3, 0.7, 0.6; γ = 1.3, rng = MersenneTwister(42))
        H2 = Q.build_hamiltonian(2, 3, 0.7, 0.6; γ = 1.3, rng = MersenneTwister(42))
        H3 = Q.build_hamiltonian(2, 3, 0.7, 0.6; γ = 1.3, rng = MersenneTwister(43))
        @test H1 == H2
        @test H1 != H3
    end

    # γ = 0 (no grain), α = 0 (no couplings), w = 0 (no field disorder) collapses
    # H to a pure diagonal of localized Sᶻ fields = hz · Σⱼ Sᶻ(site_j). We rebuild
    # that diagonal independently from the (separately tested) get_spin primitive.
    @testset "γ=0, α=0, w=0 ⇒ exact field diagonal" begin
        L_loc, L_grain, hz = 2, 2, 1.5
        S = 0.5
        B = Q.SpinBasis(S)
        states = Q.build_hilbert(L_loc + L_grain, B).states

        H = Q.build_hamiltonian(L_loc, L_grain, 1.0, 0.0;
                                γ = 0.0, w = 0.0, hz = hz, rng = MersenneTwister(9),
                                use_sparse = false)

        expected = [hz * sum(Q.get_spin(s, L_grain + j - 1, B) for j in 1:L_loc)
                    for s in states]
        @test H ≈ Diagonal(expected)
        @test H == Diagonal(diag(H))           # genuinely diagonal (no stray off-diagonals)
    end

    @testset "spin-1 (d=3) full dimension" begin
        H = Q.build_hamiltonian(1, 2, 1.0, 0.5; S = 1.0, rng = MersenneTwister(11))
        @test size(H) == (3^3, 3^3)            # d^L = 3^3 = 27
        @test ishermitian(Matrix(H))
    end
end
