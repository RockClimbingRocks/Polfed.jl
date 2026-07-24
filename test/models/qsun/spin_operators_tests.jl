using Test
using UniPolfed
using LinearAlgebra

Q = UniPolfed.Models.QSun

# Materialize an on-site spin operator as a dense (d × d) matrix by applying the
# bit-packed primitive to each single-site basis state `si ∈ 0:(d-1)` (encoded as
# state `UInt64(si)` at site 0). This lets us check the spin algebra against
# textbook ground truth rather than against a captured snapshot.
function dense_op(op, B)
    d = B.d
    M = zeros(ComplexF64, d, d)
    for si in 0:(d - 1)
        st = UInt64(si)
        if op === Q.sigma_z
            val, _ = Q.sigma_z(st, 0, B)
            M[si + 1, si + 1] += val
        else
            has_up, c_up, s_up, has_down, c_down, s_down = op(st, 0, B)
            has_up   && (M[Int(s_up) + 1, si + 1]   += c_up)
            has_down && (M[Int(s_down) + 1, si + 1] += c_down)
        end
    end
    return M
end

@testset "spin_operators" begin

    @testset "SpinBasis packing ($S)" for (S, d, bits, mask) in
                                          ((0.5, 2, 1, 0x1), (1.0, 3, 2, 0x3), (1.5, 4, 2, 0x3))
        B = Q.SpinBasis(S)
        @test B.S == S
        @test B.d == d
        @test B.bits == bits
        @test B.mask == UInt64(mask)
    end

    @testset "get_spin / set_spin round-trip ($S)" for S in (0.5, 1.0, 1.5)
        B = Q.SpinBasis(S)
        ms = collect(-S:1.0:S)
        # single site
        for m in ms
            st = Q.set_spin(UInt64(0), 0, B, m)
            @test Q.get_spin(st, 0, B) == m
        end
        # multi-site independence: setting one site must not disturb another
        st = UInt64(0)
        st = Q.set_spin(st, 0, B, ms[end])   # site 0 → +S
        st = Q.set_spin(st, 2, B, ms[1])     # site 2 → -S
        @test Q.get_spin(st, 0, B) == ms[end]
        @test Q.get_spin(st, 2, B) == ms[1]
        @test Q.get_spin(st, 1, B) == ms[1]  # untouched site stays at index 0 → -S
    end

    @testset "sigma_0 is the identity" begin
        B = Q.SpinBasis(0.5)
        for si in 0:(B.d - 1)
            st = UInt64(si)
            @test Q.sigma_0(st, 0, B) == (1.0, st)
        end
    end

    @testset "sigma_z eigenvalues ($S)" for S in (0.5, 1.0, 1.5)
        B = Q.SpinBasis(S)
        Sz = dense_op(Q.sigma_z, B)
        @test Sz ≈ Diagonal(collect(-S:1.0:S))
    end

    # The decisive ground-truth checks: the operators must close the su(2) algebra
    # and the Casimir must equal S(S+1)·I, for every spin we support.
    @testset "su(2) algebra & Casimir ($S)" for S in (0.5, 1.0, 1.5)
        B = Q.SpinBasis(S)
        Sx = dense_op(Q.sigma_x, B)
        Sy = dense_op(Q.sigma_y, B)
        Sz = dense_op(Q.sigma_z, B)

        # Hermiticity
        @test Sx ≈ Sx'
        @test Sy ≈ Sy'
        @test Sz ≈ Sz'

        # commutators [Sa, Sb] = i ε_abc Sc
        @test Sx * Sy - Sy * Sx ≈ im * Sz
        @test Sy * Sz - Sz * Sy ≈ im * Sx
        @test Sz * Sx - Sx * Sz ≈ im * Sy

        # Casimir S² = S(S+1) I
        @test Sx^2 + Sy^2 + Sz^2 ≈ (S * (S + 1)) * I(B.d)

        # ladder operator S⁺ = Sx + iSy raises only (m → m+1): strictly one
        # super/sub-diagonal, and S⁺|m_max⟩ = 0.
        Splus = Sx + im * Sy
        for c in 1:B.d, r in 1:B.d
            r == c + 1 || @test Splus[r, c] ≈ 0 atol = 1e-12
        end
    end

    @testset "spin-1/2 matches ½σ (basis ordered |down⟩,|up⟩ ⇒ m = -½,+½)" begin
        B = Q.SpinBasis(0.5)
        @test dense_op(Q.sigma_x, B) ≈ [0 0.5; 0.5 0]
        @test dense_op(Q.sigma_y, B) ≈ [0 0.5im; -0.5im 0]
        @test dense_op(Q.sigma_z, B) ≈ [-0.5 0; 0 0.5]
    end
end
