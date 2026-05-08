using LinearAlgebra
using Random
using SparseArrays

using Polfed.Models: j1j2_hamiltonian, qsun_hamiltonian, xxz_hamiltonian

@testset "Models namespace" begin
    H = qsun_hamiltonian(1, 1, 1.0, 0.5; rng=MersenneTwister(1))

    @test H isa SparseMatrixCSC
    @test size(H) == (4, 4)
end

@testset "XXZ Hamiltonian" begin
    H = xxz_hamiltonian(2, 1, 1.0, 1.0, 0.0; boundary=:open, use_sparse=false)
    expected = [
        -0.25 0.5
        0.5 -0.25
    ]

    @test H ≈ expected
    @test ishermitian(H)

    H_periodic = xxz_hamiltonian(2, 1, 1.0, 1.0, 0.0; boundary=:periodic, use_sparse=false)
    @test H_periodic ≈ 2 .* expected

    H_sparse = xxz_hamiltonian(4, 2, 1.0, 0.7, 0.0; boundary=:periodic, fields=[0.2, -0.1, 0.3, -0.4])
    H_dense = xxz_hamiltonian(
        4,
        2,
        1.0,
        0.7,
        0.0,
        boundary=:periodic,
        fields=[0.2, -0.1, 0.3, -0.4],
        use_sparse=false,
    )

    @test H_sparse isa SparseMatrixCSC
    @test size(H_sparse) == (6, 6)
    @test Matrix(H_sparse) ≈ H_dense
    @test ishermitian(H_dense)

    H_disorder_1 = xxz_hamiltonian(4, 2, 1.0, 1.0, 0.8; rng=MersenneTwister(11), use_sparse=false)
    H_disorder_2 = xxz_hamiltonian(4, 2, 1.0, 1.0, 0.8; rng=MersenneTwister(11), use_sparse=false)
    H_clean = xxz_hamiltonian(4, 2, 1.0, 1.0, 0.0; use_sparse=false)

    @test H_disorder_1 ≈ H_disorder_2
    @test H_disorder_1 != H_clean

    fields = [1.0, 2.0, -3.0]
    H_fields = xxz_hamiltonian(3, 1, 0.0, 0.0, 0.0; boundary=:open, fields=fields, use_sparse=false)

    @test H_fields ≈ Diagonal([1.0, 2.0, -3.0])

    H_explicit_1 = xxz_hamiltonian(
        4,
        2,
        1.0,
        1.0,
        3.0,
        fields=[0.4, -0.1, 0.2, -0.3],
        rng=MersenneTwister(21),
        use_sparse=false,
    )
    H_explicit_2 = xxz_hamiltonian(
        4,
        2,
        1.0,
        1.0,
        3.0,
        fields=[0.4, -0.1, 0.2, -0.3],
        rng=MersenneTwister(22),
        use_sparse=false,
    )

    @test H_explicit_1 ≈ H_explicit_2
end

@testset "J1-J2 Hamiltonian" begin
    H_j1_only = j1j2_hamiltonian(
        4,
        2,
        1.3,
        0.0,
        0.8,
        2.0,
        0.0,
        boundary=:open,
        fields=[0.1, -0.2, 0.3, -0.4],
        use_sparse=false,
    )
    H_xxz = xxz_hamiltonian(
        4,
        2,
        1.3,
        0.8,
        0.0,
        boundary=:open,
        fields=[0.1, -0.2, 0.3, -0.4],
        use_sparse=false,
    )

    @test H_j1_only ≈ H_xxz

    H_j2_open = j1j2_hamiltonian(4, 2, 0.0, 1.0, 1.0, 1.0, 0.0; boundary=:open, use_sparse=false)
    H_j2_periodic = j1j2_hamiltonian(4, 2, 0.0, 1.0, 1.0, 1.0, 0.0; boundary=:periodic, use_sparse=false)
    @test H_j2_periodic ≈ 2 .* H_j2_open

    H_j1j2 = j1j2_hamiltonian(
        6,
        3,
        1.0,
        0.5,
        1.0,
        0.6,
        0.4,
        boundary=:periodic,
        rng=MersenneTwister(12),
    )

    @test H_j1j2 isa SparseMatrixCSC
    @test size(H_j1j2) == (20, 20)
    @test ishermitian(Matrix(H_j1j2))

    H_j2_triangle = j1j2_hamiltonian(
        3,
        1,
        0.0,
        0.7,
        2.0,
        0.5,
        0.0,
        boundary=:periodic,
        use_sparse=false,
    )
    H_xxz_triangle = xxz_hamiltonian(3, 1, 0.7, 0.5, 0.0; boundary=:periodic, use_sparse=false)

    @test H_j2_triangle ≈ H_xxz_triangle

    H_sparse = j1j2_hamiltonian(
        5,
        2,
        0.9,
        -0.4,
        0.7,
        1.2,
        0.0,
        boundary=:periodic,
        fields=[0.4, -0.2, 0.1, -0.3, 0.5],
    )
    H_dense = j1j2_hamiltonian(
        5,
        2,
        0.9,
        -0.4,
        0.7,
        1.2,
        0.0,
        boundary=:periodic,
        fields=[0.4, -0.2, 0.1, -0.3, 0.5],
        use_sparse=false,
    )

    @test H_sparse isa SparseMatrixCSC
    @test size(H_sparse) == (10, 10)
    @test Matrix(H_sparse) ≈ H_dense
    @test ishermitian(H_dense)
end

@testset "Model argument validation" begin
    @test_throws ArgumentError xxz_hamiltonian(1, 0, 1.0, 1.0, 0.0)
    @test_throws ArgumentError xxz_hamiltonian(4, 2, 1.0, 1.0, 0.0; boundary=:twisted)
    @test_throws ArgumentError xxz_hamiltonian(4, -1, 1.0, 1.0, 0.0)
    @test_throws ArgumentError xxz_hamiltonian(4, 5, 1.0, 1.0, 0.0)
    @test_throws ArgumentError xxz_hamiltonian(4, 2, 1.0, 1.0, 0.0; fields=[0.0, 1.0])
    @test_throws ArgumentError xxz_hamiltonian(4, 2, 1.0, 1.0, -1.0)

    @test_throws ArgumentError j1j2_hamiltonian(1, 0, 1.0, 0.5, 1.0, 0.8, 0.0)
    @test_throws ArgumentError j1j2_hamiltonian(4, 2, 1.0, 0.5, 1.0, 0.8, 0.0; boundary=:twisted)
    @test_throws ArgumentError j1j2_hamiltonian(4, -1, 1.0, 0.5, 1.0, 0.8, 0.0)
    @test_throws ArgumentError j1j2_hamiltonian(4, 5, 1.0, 0.5, 1.0, 0.8, 0.0)
    @test_throws ArgumentError j1j2_hamiltonian(4, 2, 1.0, 0.5, 1.0, 0.8, 0.0; fields=[0.0, 1.0])
    @test_throws ArgumentError j1j2_hamiltonian(4, 2, 1.0, 0.5, 1.0, 0.8, -1.0)
end
