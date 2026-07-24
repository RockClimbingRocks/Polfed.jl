using Test
using LinearAlgebra
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

@testset "_reconstruct_eigenvectors!" begin

    @testset "matrix basis (single mul!)" begin
        # krylovbasis is n×k, ϕ is k×m: result = krylovbasis * ϕ
        n, k, m = 5, 4, 2
        Q = Matrix(qr(randn(n, k)).Q)         # orthonormal n×k basis
        ϕ = randn(k, m)                       # Ritz coefficient matrix
        expected = Q * ϕ

        vecs = zeros(n, m)
        LZ._reconstruct_eigenvectors!(LZ.ColumnLayout(), vecs, Q, ϕ)
        @test vecs ≈ expected
    end

    @testset "vector-of-vectors basis" begin
        n = 5
        e = [Float64.(1:n .== i) for i in 1:3]   # [e1, e2, e3]
        ϕ = Float64[1.0  1.0;
                    0.0  1.0;
                    0.0  0.0]                   # k=3, m=2

        vecs = zeros(n, 2)
        LZ._reconstruct_eigenvectors!(LZ.ColumnLayout(), vecs, e, ϕ)
        @test vecs[:, 1] ≈ e[1]                # 1·e1
        @test vecs[:, 2] ≈ e[1] .+ e[2]        # e1 + e2
    end

    @testset "vector-of-vectors — identity ϕ recovers basis" begin
        n = 4
        e = [Float64.(1:n .== i) for i in 1:n]
        ϕ = Matrix{Float64}(I, n, n)

        vecs = zeros(n, n)
        LZ._reconstruct_eigenvectors!(LZ.ColumnLayout(), vecs, e, ϕ)
        @test vecs ≈ Matrix{Float64}(I, n, n)
    end

    @testset "vector-of-blocks basis" begin
        n = 5; s = 2
        B1 = Matrix(qr(randn(n, s)).Q)
        B2 = Matrix(qr(randn(n, s)).Q)
        krylovbasis = [B1, B2]

        ϕ = Float64[1.0  0.0;
                    0.0  0.0;
                    0.0  1.0;
                    0.0  0.0]                  # krylovdim=4, m=2

        vecs = zeros(n, 2)
        LZ._reconstruct_eigenvectors!(LZ.ColumnLayout(), vecs, krylovbasis, ϕ)
        @test vecs[:, 1] ≈ B1[:, 1]
        @test vecs[:, 2] ≈ B2[:, 1]
    end

    @testset "zero columns selected → empty output" begin
        Q = Matrix(qr(randn(4, 3)).Q)
        ϕ = zeros(3, 0)
        vecs = zeros(4, 0)
        LZ._reconstruct_eigenvectors!(LZ.ColumnLayout(), vecs, Q, ϕ)
        @test size(vecs) == (4, 0)
    end
end

@testset "build_eigenvalues" begin
    pu = CPU()

    @testset "exact eigenvectors of a diagonal matrix → exact Rayleigh quotients" begin
        A = Diagonal([1.0, 3.0, 5.0, 7.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        vecs = Matrix{Float64}(I, 4, 4)[:, 1:3]

        vals = LZ.build_eigenvalues(LZ.ColumnLayout(),vecs, valsmap!, pu)
        @test vals ≈ [1.0, 3.0, 5.0]
        @test vals isa Vector{Float64}
    end

    @testset "complex Hermitian matrix → real eigenvalues" begin
        A = ComplexF64[2.0 1im; -1im 2.0]                # eigenvalues 1 and 3
        valsmap! = (y, x) -> mul!(y, A, x)
        v1 = normalize(ComplexF64[1.0, -1im])
        v2 = normalize(ComplexF64[1.0,  1im])
        vecs = hcat(v1, v2)

        vals = LZ.build_eigenvalues(LZ.ColumnLayout(),vecs, valsmap!, pu)
        @test sort(vals) ≈ [1.0, 3.0] atol=1e-14
        @test eltype(vals) == Float64
    end

    @testset "zero eigenvectors → returns empty vector" begin
        A = Diagonal([1.0, 2.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        @test isempty(LZ.build_eigenvalues(LZ.ColumnLayout(),zeros(2, 0), valsmap!, pu))
    end

    @testset "Float32 → result is Float32" begin
        A = Diagonal(Float32[2.0, 4.0, 6.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        vecs = Matrix{Float32}(I, 3, 3)[:, 1:2]

        vals = LZ.build_eigenvalues(LZ.ColumnLayout(),vecs, valsmap!, pu)
        @test vals ≈ Float32[2.0, 4.0]
        @test eltype(vals) == Float32
    end
end

@testset "build_residuals" begin
    pu = CPU()

    @testset "true eigenvectors → residuals ≈ 0" begin
        A = Diagonal([1.0, 3.0, 5.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        res = LZ.build_residuals(LZ.ColumnLayout(),Matrix{Float64}(I, 3, 3), [1.0, 3.0, 5.0], valsmap!, pu)
        @test res isa Vector{Float64}
        @test all(res .< 1e-14)
    end

    @testset "non-eigenvector → non-zero residual" begin
        A = Diagonal([1.0, 2.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        v = normalize([1.0, 1.0])
        vecs = reshape(v, 2, 1)
        vals = LZ.build_eigenvalues(LZ.ColumnLayout(),vecs, valsmap!, pu)   # Rayleigh quotient = 1.5
        res = LZ.build_residuals(LZ.ColumnLayout(),vecs, vals, valsmap!, pu)
        @test res[1] ≈ 0.5 atol=1e-14                     # ‖[-0.5;0.5]/√2‖
    end

    @testset "zero eigenvectors → returns empty residuals" begin
        A = Diagonal([1.0, 2.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        res = LZ.build_residuals(LZ.ColumnLayout(),zeros(2, 0), Float64[], valsmap!, pu)
        @test isempty(res)
        @test res isa Vector{Float64}
    end

    @testset "residuals always Float64 regardless of element type" begin
        A = Diagonal(Float32[1.0, 3.0])
        valsmap! = (y, x) -> mul!(y, A, x)
        res = LZ.build_residuals(LZ.ColumnLayout(),Matrix{Float32}(I, 2, 2), Float32[1.0, 3.0], valsmap!, pu)
        @test res isa Vector{Float64}
        @test all(res .< 1e-6)
    end
end

@testset "calc_eigen_residuals (combined)" begin
    # Diagonal operator: eigenvalues 1..5, eigenvectors = standard basis.
    A = Symmetric(diagm([1.0, 2.0, 3.0, 4.0, 5.0]))
    f!(y, x) = mul!(y, A, x)
    x0 = normalize(Float64[1, 0.7, -0.4, 0.9, -0.2])
    iter = LZ.LanczosIterator(f!, x0, LZ.FullRO())
    fac = LZ.lanczositer(iter, LZ.createbasis(10, x0, MatrixBasis, CPU()), 10, CPU())
    for it in 2:5
        LZ.lanczositer!(iter, fac, it)
    end

    λ, ϕ, _ = LZ.calc_ritz_residuals(fac)
    idxs = [1, 2]                                  # two lowest Ritz pairs
    vals, vecs, res = LZ.calc_eigen_residuals(fac, ϕ, idxs, f!)

    @test size(vecs) == (5, 2)
    @test sort(vals) ≈ [1.0, 2.0] atol=1e-8        # Rayleigh quotients under A
    @test all(res .< 1e-8)                         # true eigen residuals ≈ 0 at full space
    @test length(res) == 2
end
