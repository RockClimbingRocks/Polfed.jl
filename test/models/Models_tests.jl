using Test
using Polfed
using LinearAlgebra
using SparseArrays
using Random

using Polfed.Models: qsun_hamiltonian, xxz_hamiltonian, j1j2_hamiltonian

@testset "Models namespace" begin

    @testset "exports are reachable" begin
        @test qsun_hamiltonian isa Function
        @test xxz_hamiltonian isa Function
        @test j1j2_hamiltonian isa Function
    end

    @testset "qsun_hamiltonian wrapper forwards to QSun" begin
        # the typed Models wrapper must agree with the underlying QSun method
        H = qsun_hamiltonian(2, 2, 1.0, 0.5; rng = MersenneTwister(1))
        ref = Polfed.Models.QSun.qsun_hamiltonian(2, 2, 1.0, 0.5; rng = MersenneTwister(1))
        @test H == ref
        @test size(H) == (16, 16)

        Hu = qsun_hamiltonian(2, 2, 1.0, 0.5; use_U1 = true, S_z = 0.0, rng = MersenneTwister(2))
        refu = Polfed.Models.QSun.qsun_hamiltonian(2, 2, 1.0, 0.5; use_U1 = true, S_z = 0.0,
                                                   rng = MersenneTwister(2))
        @test Hu == refu
        @test size(Hu) == (6, 6)
    end

    @testset "return types follow use_sparse" begin
        @test qsun_hamiltonian(1, 1, 1.0, 0.5; rng = MersenneTwister(1)) isa SparseMatrixCSC
        @test qsun_hamiltonian(1, 1, 1.0, 0.5; use_sparse = false,
                               rng = MersenneTwister(1)) isa Matrix{Float64}
    end
end
