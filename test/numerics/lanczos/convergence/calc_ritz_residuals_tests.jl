using Test
using LinearAlgebra
using Random
using UniPolfed
using UniPolfed.Numerics.Lanczos

LZ = UniPolfed.Numerics.Lanczos

@testset "calculateresidual" begin

    @testset "scalar β — basic formula abs(β · ϕ[end, i])" begin
        # Two-vector Krylov space, two Ritz vectors as columns.
        ϕ = Float64[0.6 0.0;
                    0.8 1.0]    # ϕ[end, 1] = 0.8, ϕ[end, 2] = 1.0

        @test LZ.calculateresidual(ϕ, 2.0, 1) ≈ abs(2.0 * 0.8)
        @test LZ.calculateresidual(ϕ, 2.0, 2) ≈ abs(2.0 * 1.0)
        @test LZ.calculateresidual(ϕ, 0.0, 1) == 0.0   # exact breakdown → zero residual
    end

    @testset "scalar β — negative coupling (sign irrelevant)" begin
        ϕ = Float64[-0.8; 0.6][:, :]
        @test LZ.calculateresidual(ϕ, -3.0, 1) ≈ abs(-3.0 * 0.6)
    end

    @testset "scalar β — complex element type" begin
        ϕ = ComplexF64[0.6+0im; 0.8+0im][:, :]
        # real β, complex ϕ: result should be real-valued
        r = LZ.calculateresidual(ϕ, 2.0, 1)
        @test r ≈ 1.6
        @test r isa Float64
    end

    @testset "block β — formula norm(β · ϕ[end-s+1:end, i])" begin
        # s=2, krylovdim=4 → trailing 2 rows are ϕ[3:4, :]
        ϕ = Float64[1.0  0.0;
                    0.0  1.0;
                    0.6  0.0;    # row 3 (end-1)
                    0.8  0.5]    # row 4 (end)

        β = Diagonal([1.0, 2.0])   # 2×2 block coupling

        # column 1: β * [0.6; 0.8] = [0.6; 1.6]
        @test LZ.calculateresidual(ϕ, Matrix(β), 1) ≈ norm([0.6, 1.6])
        # column 2: β * [0.0; 0.5] = [0.0; 1.0]
        @test LZ.calculateresidual(ϕ, Matrix(β), 2) ≈ norm([0.0, 1.0])
    end

    @testset "block β — zero coupling gives zero residual" begin
        ϕ = ones(4, 2)
        β = zeros(2, 2)
        @test LZ.calculateresidual(ϕ, β, 1) == 0.0
        @test LZ.calculateresidual(ϕ, β, 2) == 0.0
    end
end

@testset "calc_ritz_residuals (full factorization)" begin
    # Diagonal operator with eigenvalues 1..5 and standard-basis eigenvectors.
    A = Symmetric(diagm([1.0, 2.0, 3.0, 4.0, 5.0]))
    f!(y, x) = mul!(y, A, x)
    x0 = normalize(Float64[1, 0.7, -0.4, 0.9, -0.2])
    iter = LZ.LanczosIterator(f!, x0, LZ.FullRO())
    fac = LZ.lanczositer(iter, LZ.createbasis(10, x0, MatrixBasis, CPU()), 10, CPU())

    for it in 2:5
        LZ.lanczositer!(iter, fac, it)
    end

    λ, ϕ, residuals = LZ.calc_ritz_residuals(fac)
    @test length(residuals) == fac.krylovdim == length(λ)   # one residual per Ritz pair
    @test issorted(λ)                                        # ascending Ritz values
    @test λ ≈ [1.0, 2.0, 3.0, 4.0, 5.0]                      # full space → exact spectrum
    @test all(residuals .>= 0)
end

function _partial_factorization(A, x0, layout; nsteps=3)
    seed = LZ._prepare_seed(layout, x0)
    f! = layout isa RowLayout ?
         ((Y, X) -> mul!(Y, X, A)) :
         ((Y, X) -> mul!(Y, A, X))
    iterator = LZ.LanczosIterator(f!, seed, FullRO(), layout)
    basis = LZ.createbasis(32, seed, MatrixBasis, CPU(), layout)
    factorization = LZ.lanczositer(iterator, basis, 32, CPU())
    for iteration in 2:(nsteps + 1)
        LZ.lanczositer!(iterator, factorization, iteration) || break
    end
    return factorization
end

function _explicit_ritz_residuals(A, factorization, λ, ϕ)
    basis = Matrix(LZ.allvecs(factorization.basis))
    if factorization.layout isa RowLayout
        vectors = ϕ' * basis
        return [
            norm(A * collect(view(vectors, i, :)) - λ[i] * collect(view(vectors, i, :)))
            for i in eachindex(λ)
        ]
    end

    vectors = basis * ϕ
    return [
        norm(A * view(vectors, :, i) - λ[i] * view(vectors, :, i))
        for i in eachindex(λ)
    ]
end

@testset "current Ritz coupling matches explicit residuals" begin
    Random.seed!(20260721)
    n = 12
    raw = randn(n, n)
    A = Matrix(Symmetric(raw + raw'))

    scalar = _partial_factorization(A, randn(n), ColumnLayout(); nsteps=5)
    λs, ϕs, rs = LZ.calc_ritz_residuals(scalar)
    @test rs ≈ _explicit_ritz_residuals(A, scalar, λs, ϕs) atol=1e-11 rtol=1e-10

    x0 = randn(n, 2)
    columns = _partial_factorization(A, x0, ColumnLayout(); nsteps=3)
    λc, ϕc, rc = LZ.calc_ritz_residuals(columns)
    @test rc ≈ _explicit_ritz_residuals(A, columns, λc, ϕc) atol=1e-11 rtol=1e-10

    rows = _partial_factorization(A, permutedims(x0), RowLayout(); nsteps=3)
    λr, ϕr, rr = LZ.calc_ritz_residuals(rows)
    @test rr ≈ _explicit_ritz_residuals(A, rows, λr, ϕr) atol=1e-11 rtol=1e-10
end

@testset "full-dimensional breakdown uses zero current coupling" begin
    Random.seed!(7)
    A = Diagonal(collect(1.0:5.0))
    f!(y, x) = mul!(y, A, x)
    result = lanczos(
        f!, randn(5), LowestEigs(2);
        maxdim=5, tol=1e-10, eigentol=1e-8,
    )

    @test collect(result.vals) ≈ [1.0, 2.0] atol=1e-10
    @test result.report.complete
    @test result.report.residual == 0.0
end
