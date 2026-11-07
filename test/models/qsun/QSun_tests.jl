using Test
using Polfed
using LinearAlgebra
using SparseArrays
using Random

Q = Polfed.Models.QSun

@testset "qsun_hamiltonian (dispatch wrapper)" begin

    @testset "use_U1=false ⇒ full-space build_hamiltonian" begin
        H = Q.qsun_hamiltonian(2, 2, 1.0, 0.5; rng = MersenneTwister(1))
        ref = Q.build_hamiltonian(2, 2, 1.0, 0.5; rng = MersenneTwister(1))
        @test size(H) == (16, 16)
        @test H == ref                          # wrapper forwards verbatim
    end

    @testset "use_U1=true ⇒ sector build_hamiltonian_cons" begin
        H = Q.qsun_hamiltonian(2, 2, 1.0, 0.5; use_U1 = true, S_z = 0.0,
                               rng = MersenneTwister(2))
        ref = Q.build_hamiltonian_cons(2, 2, 1.0, 0.5; S_z = 0.0, rng = MersenneTwister(2))
        @test size(H) == (6, 6)
        @test H == ref
    end

    @testset "return types follow use_sparse" begin
        @test Q.qsun_hamiltonian(1, 1, 1.0, 0.5; rng = MersenneTwister(1)) isa SparseMatrixCSC
        @test Q.qsun_hamiltonian(1, 1, 1.0, 0.5; use_sparse = false,
                                 rng = MersenneTwister(1)) isa Matrix{Float64}
    end

    @testset "rejects non-(half-)integer spin" begin
        @test_throws ErrorException Q.qsun_hamiltonian(1, 1, 1.0, 0.5; S = 0.3)
        # valid half-integer spin must NOT throw
        @test Q.qsun_hamiltonian(1, 1, 1.0, 0.5; S = 1.5, rng = MersenneTwister(1)) isa SparseMatrixCSC
    end
end
