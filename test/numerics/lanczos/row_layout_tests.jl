using Test
using Polfed
using Polfed.Numerics.Lanczos
using LinearAlgebra
using Random

# Row-native Lanczos: driving the factorization in RowLayout (bs×D blocks, operator X·H) must
# return the SAME eigenpairs as the default ColumnLayout (D×bs, H·X) on the same Hermitian
# operator — the block storage orientation is a pure implementation detail. This is the
# end-to-end oracle for the Layout generalization.

@testset "RowLayout vs ColumnLayout (block Lanczos)" begin
    Random.seed!(11)
    n = 60
    # isolated extremes so the targeted eigenpairs converge well below the full space
    spec = sort(vcat([-10.0, -8.0, -6.0, 6.0, 8.0, 10.0], range(-1.0, 1.0; length = n - 6)))

    for T in (Float64, ComplexF64)
        Q = Matrix(qr(randn(T, n, n)).Q)
        H = Matrix(Hermitian(Q * Diagonal(spec) * Q'))
        f_col!(Y, X) = mul!(Y, H, X)          # column: Y = H·X   (X is D×bs)
        f_row!(Y, X) = mul!(Y, X, H)          # row:    Y = X·H   (X is bs×D)
        exact = real(eigen(H).values)         # ascending

        bs = 2
        x0c = randn(T, n, bs)
        x0r = collect(transpose(x0c))         # same seed, transposed to bs×n

        for target in (LowestEigs(3), HighestEigs(3))
            rc = lanczos(f_col!, x0c, target; maxdim = 36)
            rr = lanczos(f_row!, x0r, target; maxdim = 36, layout = RowLayout())

            want = target isa Polfed.Numerics.Lanczos.Lowest ? exact[1:3] : exact[end-2:end]

            # both converge the same count and the correct values
            @test rc.converged == 3
            @test rr.converged == 3
            @test isapprox(sort(real(collect(rc.vals))), want; atol = 1e-7)
            @test isapprox(sort(real(collect(rr.vals))), want; atol = 1e-7)
            # row == column (the invariant)
            @test isapprox(sort(real(collect(rr.vals))), sort(real(collect(rc.vals))); atol = 1e-8)

            # eigenvectors are stored as ROWS in row layout (nconv × D) and satisfy A·v = λ·v
            @test size(rr.vecs) == (3, n)
            @test size(rc.vecs) == (n, 3)
            # RowLayout stores its maxdim-sized basis capacity on axis 1, not the
            # operator-dimension axis. The report must still express this as columns.
            @test rr.report.ncols_cpu_reserved == 36
            @test rr.report.ncols_cpu_needed <= rr.report.ncols_cpu_reserved
            @test all(rr.residuals .< 1e-6)
            for k in 1:3
                v = collect(rr.vecs[k, :])
                λ = real(collect(rr.vals))[k]
                # row eigenvectors are LEFT eigenvectors: the row v satisfies v·H = λ·v,
                # i.e. transpose(H)·v = λ·v (= H·v only in the real-symmetric case).
                @test norm(transpose(H) * v - λ * v) < 1e-6
            end
        end
    end
end

@testset "RowLayout rejects unsupported VectorBasis at the public boundary" begin
    x0 = randn(2, 8)
    passthrough!(Y, X) = copyto!(Y, X)

    err = try
        lanczos(
            passthrough!, x0, LowestEigs(1);
            layout=RowLayout(), basistype=VectorBasis, maxdim=4,
        )
        nothing
    catch caught
        caught
    end

    @test err isa ArgumentError
    @test occursin(
        "RowLayout with VectorBasis is not supported",
        sprint(showerror, err),
    )
end
