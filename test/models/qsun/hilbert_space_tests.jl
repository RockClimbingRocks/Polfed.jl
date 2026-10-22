using Test
using UniPolfed

Q = UniPolfed.Models.QSun

@testset "hilbert_space" begin

    @testset "build_hilbert — spin-1/2 enumerates 0:2^L-1" begin
        B = Q.SpinBasis(0.5)
        for L in 1:6
            H = Q.build_hilbert(L, B)
            @test H.L == L
            @test length(H.states) == 2^L
            @test issorted(H.states)
            @test allunique(H.states)
            @test H.states == UInt64.(0:(2^L - 1))    # every bit pattern is a valid state
        end
    end

    @testset "build_hilbert — spin-1 (d=3) cardinality & packing" begin
        B = Q.SpinBasis(1.0)              # d = 3, bits = 2
        H = Q.build_hilbert(2, B)
        @test length(H.states) == 3^2     # d^L
        @test allunique(H.states)
        @test issorted(H.states)
        # two sites, each si ∈ 0:2 packed at bit-strides 0 and 2 ⇒ si1*4 + si0
        expected = sort(UInt64[si1 * 4 + si0 for si1 in 0:2 for si0 in 0:2])
        @test H.states == expected
    end

    @testset "state_index — round-trip and out-of-space sentinel" begin
        B = Q.SpinBasis(0.5)
        H = Q.build_hilbert(4, B)
        for (k, s) in enumerate(H.states)
            @test Q.state_index(H, s) == k
        end
        # a state beyond the enumerated range is reported as 0 (not found)
        @test Q.state_index(H, UInt64(2^4)) == 0

        # spin-1: an invalid local payload (si = 3 ∉ 0:2) is absent ⇒ 0
        B1 = Q.SpinBasis(1.0)
        H1 = Q.build_hilbert(2, B1)
        @test Q.state_index(H1, UInt64(3)) == 0   # site 0 payload = 3 is invalid
    end
end
