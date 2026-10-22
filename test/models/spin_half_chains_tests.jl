using Test
using Polfed
using LinearAlgebra
using SparseArrays
using Random

using Polfed.Models: xxz_hamiltonian, j1j2_hamiltonian

# ── Independent ground-truth XXZ builder ──────────────────────────────────────
# Build the full-space Hamiltonian from kron products of the standard spin-1/2
# matrices, then restrict to the fixed-Lup sector. This shares no code with the
# package's push!-based assembly, so agreement is a genuine cross-check.
# Convention: state index = state + 1, bit (i-1) = site i, set bit ⇒ up (Sᶻ=+½).
const _sx = ComplexF64[0 0.5; 0.5 0]
const _sy = ComplexF64[0 -0.5im; 0.5im 0]
const _sz = ComplexF64[-0.5 0; 0 0.5]
const _i2 = ComplexF64[1 0; 0 1]

function _op_at(op, i, L)
    out = (L == i) ? op : _i2
    for k in (L - 1):-1:1            # site 1 ends up least-significant (bit 0)
        out = kron(out, (k == i) ? op : _i2)
    end
    return out
end

function _brute_xxz(L, Lup, J, Δ, fields; boundary)
    bonds = boundary === :open ? [(i, i + 1) for i in 1:(L - 1)] :
            [(i, mod1(i + 1, L)) for i in 1:L if i != mod1(i + 1, L)]
    H = zeros(ComplexF64, 2^L, 2^L)
    for (i, j) in bonds
        H .+= J .* (_op_at(_sx, i, L) * _op_at(_sx, j, L) .+
                    _op_at(_sy, i, L) * _op_at(_sy, j, L) .+
                    Δ .* _op_at(_sz, i, L) * _op_at(_sz, j, L))
    end
    for i in 1:L
        H .+= fields[i] .* _op_at(_sz, i, L)
    end
    idx = [s + 1 for s in 0:(2^L - 1) if count_ones(s) == Lup]
    sector = H[idx, idx]
    @assert norm(imag(sector)) < 1e-12      # XXZ is real in this basis
    return real(sector)
end

@testset "spin_half_chains" begin

    @testset "Heisenberg dimer — exact spectrum" begin
        H = xxz_hamiltonian(2, 1, 1.0, 1.0, 0.0; boundary = :open, use_sparse = false)
        @test H ≈ [-0.25 0.5; 0.5 -0.25]
        @test sort(eigvals(Symmetric(H))) ≈ [-0.75, 0.25]   # singlet / triplet
        # periodic L=2 lists each bond from both ends ⇒ exactly doubled
        Hp = xxz_hamiltonian(2, 1, 1.0, 1.0, 0.0; boundary = :periodic, use_sparse = false)
        @test Hp ≈ 2 .* H
    end

    @testset "field-only limit is diagonal" begin
        H = xxz_hamiltonian(3, 1, 0.0, 0.0, 0.0; boundary = :open,
                            fields = [1.0, 2.0, -3.0], use_sparse = false)
        @test H ≈ Diagonal([1.0, 2.0, -3.0])
    end

    @testset "matches independent kron construction" begin
        cases = [
            (4, 2, 1.0, 0.7, [0.2, -0.1, 0.3, -0.4], :open),
            (4, 2, 1.0, 0.7, [0.2, -0.1, 0.3, -0.4], :periodic),
            (3, 1, 2.0, 0.5, [0.3, -0.7, 1.1], :open),
            (5, 2, -0.8, 1.3, [0.1, 0.2, -0.3, 0.4, -0.5], :periodic),
        ]
        for (L, Lup, J, Δ, fields, bc) in cases
            H = xxz_hamiltonian(L, Lup, J, Δ, 0.0; boundary = bc,
                                fields = fields, use_sparse = false)
            @test H ≈ _brute_xxz(L, Lup, J, Δ, fields; boundary = bc)
        end
    end

    @testset "hardcoded reference (3-site open)" begin
        H = xxz_hamiltonian(3, 1, 2.0, 0.5, 0.0; boundary = :open,
                            fields = [0.3, -0.7, 1.1], use_sparse = false)
        @test H ≈ [-0.05 1.0 0.0; 1.0 -1.55 1.0; 0.0 1.0 0.75]
    end

    @testset "dimensions, hermiticity, sparse/dense, reproducibility" begin
        for (L, Lup) in ((4, 2), (6, 3), (5, 2))
            H = xxz_hamiltonian(L, Lup, 1.0, 0.7, 0.0)
            @test size(H) == (binomial(L, Lup), binomial(L, Lup))
            @test ishermitian(Matrix(H))
        end
        Hs = xxz_hamiltonian(4, 2, 1.0, 0.7, 0.0; fields = [0.2, -0.1, 0.3, -0.4])
        Hd = xxz_hamiltonian(4, 2, 1.0, 0.7, 0.0; fields = [0.2, -0.1, 0.3, -0.4],
                             use_sparse = false)
        @test Hs isa SparseMatrixCSC
        @test Matrix(Hs) ≈ Hd
        # disorder is RNG-driven but reproducible from a seed
        a = xxz_hamiltonian(4, 2, 1.0, 1.0, 0.8; rng = MersenneTwister(11), use_sparse = false)
        b = xxz_hamiltonian(4, 2, 1.0, 1.0, 0.8; rng = MersenneTwister(11), use_sparse = false)
        @test a ≈ b
        @test a != xxz_hamiltonian(4, 2, 1.0, 1.0, 0.0; use_sparse = false)  # disorder shifts it
    end

    @testset "validation errors" begin
        @test_throws ArgumentError xxz_hamiltonian(1, 0, 1.0, 1.0, 0.0)
        @test_throws ArgumentError xxz_hamiltonian(4, 2, 1.0, 1.0, 0.0; boundary = :twisted)
        @test_throws ArgumentError xxz_hamiltonian(4, -1, 1.0, 1.0, 0.0)
        @test_throws ArgumentError xxz_hamiltonian(4, 5, 1.0, 1.0, 0.0)
        @test_throws ArgumentError xxz_hamiltonian(4, 2, 1.0, 1.0, 0.0; fields = [0.0, 1.0])
        @test_throws ArgumentError xxz_hamiltonian(4, 2, 1.0, 1.0, -1.0)
    end
end

@testset "j1j2 reduces to / extends xxz" begin

    @testset "J2 = 0 ⇒ identical to xxz (any Δ2)" begin
        H_j = j1j2_hamiltonian(4, 2, 1.3, 0.0, 0.8, 2.0, 0.0; boundary = :open,
                               fields = [0.1, -0.2, 0.3, -0.4], use_sparse = false)
        H_x = xxz_hamiltonian(4, 2, 1.3, 0.8, 0.0; boundary = :open,
                              fields = [0.1, -0.2, 0.3, -0.4], use_sparse = false)
        @test H_j ≈ H_x
    end

    @testset "triangle identity (L=3 periodic): J2 bond ≡ nearest-neighbor" begin
        # on a 3-site ring, next-nearest neighbors wrap onto nearest neighbors
        H_j2 = j1j2_hamiltonian(3, 1, 0.0, 0.7, 2.0, 0.5, 0.0; boundary = :periodic,
                                use_sparse = false)
        H_x = xxz_hamiltonian(3, 1, 0.7, 0.5, 0.0; boundary = :periodic, use_sparse = false)
        @test H_j2 ≈ H_x
    end

    @testset "J1=0: next-nearest channel doubles under periodic L=4" begin
        Ho = j1j2_hamiltonian(4, 2, 0.0, 1.0, 1.0, 1.0, 0.0; boundary = :open, use_sparse = false)
        Hp = j1j2_hamiltonian(4, 2, 0.0, 1.0, 1.0, 1.0, 0.0; boundary = :periodic, use_sparse = false)
        # the next-nearest graph of a 4-ring is two 2-site bonds, each listed from
        # both ends by the periodic enumeration ⇒ exactly doubled
        @test Hp ≈ 2 .* Ho
    end

    @testset "hardcoded reference (4-site open)" begin
        H = j1j2_hamiltonian(4, 2, 1.2, -0.8, 0.5, 1.5, 0.0; boundary = :open,
                             fields = [0.2, -0.4, 0.6, -0.8], use_sparse = false)
        expected = [
            0.75   0.6  -0.4  -0.4   0.0   0.0
            0.6   -0.05  0.6   0.6   0.0   0.0
           -0.4    0.6   0.85  0.0   0.6  -0.4
           -0.4    0.6   0.0   0.05  0.6  -0.4
            0.0    0.0   0.6   0.6  -2.05  0.6
            0.0    0.0  -0.4  -0.4   0.6   0.75
        ]
        @test H ≈ expected
    end

    @testset "dimensions, hermiticity, validation" begin
        H = j1j2_hamiltonian(6, 3, 1.0, 0.5, 1.0, 0.6, 0.4; boundary = :periodic,
                             rng = MersenneTwister(12))
        @test H isa SparseMatrixCSC
        @test size(H) == (binomial(6, 3), binomial(6, 3))
        @test ishermitian(Matrix(H))

        @test_throws ArgumentError j1j2_hamiltonian(1, 0, 1.0, 0.5, 1.0, 0.8, 0.0)
        @test_throws ArgumentError j1j2_hamiltonian(4, 2, 1.0, 0.5, 1.0, 0.8, 0.0; boundary = :twisted)
        @test_throws ArgumentError j1j2_hamiltonian(4, 5, 1.0, 0.5, 1.0, 0.8, 0.0)
        @test_throws ArgumentError j1j2_hamiltonian(4, 2, 1.0, 0.5, 1.0, 0.8, 0.0; fields = [0.0, 1.0])
        @test_throws ArgumentError j1j2_hamiltonian(4, 2, 1.0, 0.5, 1.0, 0.8, -1.0)
    end
end
