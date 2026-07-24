using Test
using LinearAlgebra
using Random
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

# A `bs`-vector block in the given layout: D×bs (column) or bs×D (row).
makeblock(::ColumnLayout, ::Type{T}, D, bs) where {T} = randn(T, D, bs)
makeblock(::RowLayout,    ::Type{T}, D, bs) where {T} = randn(T, bs, D)

@testset "Layout primitives" begin
    D, bs = 9, 3

    @testset "$(nameof(typeof(L))), $T" for L in (ColumnLayout(), RowLayout()),
                                              T in (Float32, Float64)
        Random.seed!(7)
        tol = T == Float32 ? 1f-4 : 1e-11

        @testset "blocksize / opdim" begin
            X = makeblock(L, T, D, bs)
            @test LZ.blocksize(L, X) == bs
            @test LZ.opdim(L, X) == D
        end

        @testset "orthonormalize: Q orthonormal and X = Q·β / β·Q" begin
            X = makeblock(L, T, D, bs)
            Q, β = LZ.orthonormalize(L, X)
            @test size(Q) == size(X)
            @test size(β) == (bs, bs)
            # Q has orthonormal vectors → its self-overlap is the identity
            @test isapprox(LZ.overlap(L, Q, Q), Matrix{T}(I, bs, bs); atol = tol)
            # factorization reproduces X (Q·β column, β·Q row)
            recon = L isa ColumnLayout ? Q * β : β * Q
            @test isapprox(recon, X; atol = tol)
        end

        @testset "overlap + project! ⇒ residual orthogonal to V" begin
            V, _ = LZ.orthonormalize(L, makeblock(L, T, D, bs))   # orthonormal block
            R = makeblock(L, T, D, bs)
            α = LZ.overlap(L, V, R)
            @test size(α) == (bs, bs)
            LZ.project!(L, R, V, α)
            @test norm(LZ.overlap(L, V, R)) < tol                 # R now ⟂ V
        end

        @testset "reorthogonalize! ⇒ W orthogonal to the basis" begin
            k = 4
            Q, _ = LZ.orthonormalize(L, makeblock(L, T, D, k))     # k-vector orthonormal basis
            W = makeblock(L, T, D, bs)
            LZ.reorthogonalize!(L, W, Q)
            @test norm(LZ.overlap(L, Q, W)) < tol                  # W ⟂ span(Q)
        end

        @testset "reconstruct: selects basis vectors via ϕ" begin
            k, m = 5, 2
            basis, _ = LZ.orthonormalize(L, makeblock(L, T, D, k))
            ϕ = zeros(T, k, m)
            ϕ[2, 1] = one(T)          # pick basis vector 2
            ϕ[4, 2] = one(T)          # pick basis vector 4
            V = LZ.reconstruct(L, basis, ϕ)
            if L isa ColumnLayout
                @test size(V) == (D, m)
                @test isapprox(V[:, 1], basis[:, 2]; atol = tol)
                @test isapprox(V[:, 2], basis[:, 4]; atol = tol)
            else
                @test size(V) == (m, D)
                @test isapprox(V[1, :], basis[2, :]; atol = tol)
                @test isapprox(V[2, :], basis[4, :]; atol = tol)
            end
        end
    end

    # ── Row layout is the transpose of column layout (real, symmetric setting) ──
    @testset "row ≡ column transposed" begin
        Random.seed!(11)
        T = Float64
        Vc = randn(T, D, bs);  Vr = permutedims(Vc)
        Rc = randn(T, D, bs);  Rr = permutedims(Rc)

        # overlap(Row) is overlap(Column) transposed
        @test LZ.overlap(RowLayout(), Vr, Rr) ≈ permutedims(LZ.overlap(ColumnLayout(), Vc, Rc))

        # reconstruct gives transposed-equal eigenvectors
        k, m = 5, 2
        Bc = randn(T, D, k); Br = permutedims(Bc)
        ϕ = randn(T, k, m)
        @test permutedims(LZ.reconstruct(RowLayout(), Br, ϕ)) ≈ LZ.reconstruct(ColumnLayout(), Bc, ϕ)

        # project! produces transposed-equal residuals
        Qc, _ = LZ.orthonormalize(ColumnLayout(), randn(T, D, bs)); Qr = permutedims(Qc)
        Wc = randn(T, D, bs); Wr = permutedims(Wc)
        LZ.project!(ColumnLayout(), Wc, Qc, LZ.overlap(ColumnLayout(), Qc, Wc))
        LZ.project!(RowLayout(),    Wr, Qr, LZ.overlap(RowLayout(),    Qr, Wr))
        @test permutedims(Wr) ≈ Wc
    end
end
