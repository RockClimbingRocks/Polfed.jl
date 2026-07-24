using Test
using UniPolfed
using LinearAlgebra
using SparseArrays
using Random

Q = UniPolfed.Models.QSun

@testset "quantum_sun_u1_conserving (build_hamiltonian_cons)" begin

    @testset "hermiticity, types, reproducibility" begin
        H = Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 0.0, rng = MersenneTwister(1))
        @test H isa SparseMatrixCSC
        @test ishermitian(Matrix(H))

        Hd = Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 0.0,
                                      rng = MersenneTwister(1), use_sparse = false)
        @test Hd isa Matrix{Float64}
        @test Hd ≈ Matrix(H)

        H2 = Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 0.0, rng = MersenneTwister(7))
        H3 = Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 0.0, rng = MersenneTwister(7))
        @test H2 == H3
    end

    # The defining property of the U(1) builder: each fixed-Sᶻ sector has dimension
    # binomial(L, n_up), and the sectors partition the full 2^L Hilbert space.
    @testset "sector dimensions tile the full space (L=4, S=1/2)" begin
        L_loc, L_grain = 2, 2
        L = L_loc + L_grain
        total = 0
        for S_z in -2.0:1.0:2.0
            H = Q.build_hamiltonian_cons(L_loc, L_grain, 1.0, 0.5; S_z = S_z,
                                         rng = MersenneTwister(3))
            n_up = Int(S_z + L / 2)
            @test size(H, 1) == binomial(L, n_up)
            total += size(H, 1)
        end
        @test total == 2^L
    end

    @testset "higher spin sector (S=3/2)" begin
        # two spin-3/2 sites, total Sᶻ = 1 ⇒ {(3/2,-1/2),(-1/2,3/2),(1/2,1/2)} = 3 states
        H = Q.build_hamiltonian_cons(1, 1, 1.0, 0.5; S = 1.5, S_z = 1.0,
                                     rng = MersenneTwister(4))
        @test size(H) == (3, 3)
        @test ishermitian(Matrix(H))
    end

    @testset "invalid sector is rejected" begin
        # non-integer (S_z + L·S): 0.3 + 4·0.5 = 2.3 ∉ ℤ
        @test_throws ErrorException Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 0.3)
        # |S_z| exceeds L·S = 2
        @test_throws ErrorException Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 3.0)
    end
end
