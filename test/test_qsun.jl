using LinearAlgebra
using Random
using SparseArrays

@testset "QSun full Hilbert-space Hamiltonian" begin
    rng_sparse = MersenneTwister(1234)
    H_sparse = Polfed.QSun.qsun_hamiltonian(2, 2, 1.0, 0.5; rng=rng_sparse)

    @test H_sparse isa SparseMatrixCSC
    @test size(H_sparse) == (16, 16)
    @test ishermitian(Matrix(H_sparse))

    rng_dense = MersenneTwister(1234)
    H_dense = Polfed.QSun.qsun_hamiltonian(2, 2, 1.0, 0.5; use_sparse=false, rng=rng_dense)

    @test H_dense isa Matrix
    @test H_dense ≈ Matrix(H_sparse)
end

@testset "QSun U(1)-conserving Hamiltonian" begin
    rng = MersenneTwister(4321)
    H_u1 = Polfed.QSun.qsun_hamiltonian(
        2,
        2,
        1.0,
        0.5;
        use_U1=true,
        S_z=0.0,
        rng=rng,
    )

    @test H_u1 isa SparseMatrixCSC
    @test size(H_u1) == (6, 6)
    @test ishermitian(Matrix(H_u1))

    H_dense = Polfed.QSun.qsun_hamiltonian(
        2,
        2,
        1.0,
        0.5;
        use_U1=true,
        S_z=0.0,
        use_sparse=false,
        rng=MersenneTwister(4321),
    )

    @test H_dense isa Matrix
    @test H_dense ≈ Matrix(H_u1)

    H_half_integer_sector = Polfed.QSun.qsun_hamiltonian(
        2,
        1,
        1.0,
        0.5;
        use_U1=true,
        S_z=0.5,
        rng=MersenneTwister(12),
    )

    @test size(H_half_integer_sector) == (3, 3)
    @test ishermitian(Matrix(H_half_integer_sector))

    H_spin_three_half = Polfed.QSun.qsun_hamiltonian(
        1,
        1,
        1.0,
        0.5;
        S=1.5,
        use_U1=true,
        S_z=1.0,
        rng=MersenneTwister(34),
    )

    @test size(H_spin_three_half) == (3, 3)
    @test ishermitian(Matrix(H_spin_three_half))
end
