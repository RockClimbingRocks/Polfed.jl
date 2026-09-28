using Test
using UniPolfed
using LinearAlgebra
using Random
using SparseArrays

K = UniPolfed.Numerics.KPM

# Closed-form Chebyshev value, |x| ≤ 1.
_kpm_test_cheb(x, k) = cos(k * acos(x))

# Matrix-free operator (mul!/size/eltype) — the LinearOperators.jl contract.
struct MomTestOp
    M::Matrix{Float64}
end
LinearAlgebra.mul!(y, op::MomTestOp, x) = mul!(y, op.M, x)
Base.size(op::MomTestOp, d) = size(op.M, d)
Base.eltype(::MomTestOp) = Float64


@testset "moments" begin

    @testset "trace_doubling! basis probe gives exact T_k(lambda)" begin
        λ = [0.3, -0.7, 0.5, -0.1]
        H = Diagonal(λ)
        f!(y, x) = mul!(y, H, x)

        N = 8
        for j in eachindex(λ)
            probe = zeros(length(λ))
            probe[j] = 1.0

            μ = zeros(N)
            work = K.ChebyshevWorkspace(probe)

            K.trace_doubling!(f!, probe, μ, work)

            expected = [_kpm_test_cheb(λ[j], k) for k in 0:N-1]
            @test μ ≈ expected
        end
    end

    @testset "trace_doubling! full basis gives Tr T_k(H)" begin
        λ = [0.3, -0.7, 0.5, -0.1]
        H = Diagonal(λ)
        f!(y, x) = mul!(y, H, x)

        N = 10
        μ = zeros(N)

        for j in eachindex(λ)
            probe = zeros(length(λ))
            probe[j] = 1.0
            work = K.ChebyshevWorkspace(probe)

            K.trace_doubling!(f!, probe, μ, work)
        end

        expected = [sum(_kpm_test_cheb(λj, k) for λj in λ) for k in 0:N-1]
        @test μ ≈ expected
    end

    @testset "trace_doubling! supports column probe blocks" begin
        λ = [0.3, -0.7, 0.5, -0.1]
        H = Diagonal(λ)
        f!(Y, X) = mul!(Y, H, X)

        N = 8
        probe = zeros(length(λ), 2)
        probe[1, 1] = 1.0
        probe[3, 2] = 1.0
        μ = zeros(N)
        work = K.ChebyshevWorkspace(probe)

        K.trace_doubling!(f!, probe, μ, work)

        expected = [_kpm_test_cheb(λ[1], k) + _kpm_test_cheb(λ[3], k) for k in 0:N-1]
        @test μ ≈ expected
    end

    @testset "probe_moments! keeps each probe in its own column" begin
        λ = [0.3, -0.7, 0.5, -0.1]
        H = Diagonal(λ)
        N = 9
        for (probe, f!) in ((Matrix{Float64}(I, 4, 4), (Y, X) -> mul!(Y, H, X)),     # columns
                            (Matrix{Float64}(I, 4, 4)[1:3, :], (Y, X) -> mul!(Y, X, H)))  # rows
            μ = zeros(N, size(probe, 1) == 3 ? 3 : 4)
            K.probe_moments!(μ, f!, probe, K.ChebyshevWorkspace(probe))
            for p in axes(μ, 2)
                @test μ[:, p] ≈ [_kpm_test_cheb(λ[p], k) for k in 0:N-1]
            end
        end
    end

    @testset "block width and ntasks change neither the probes nor the moments" begin
        # Probes are drawn one at a time and summed in probe order, so with an operator whose
        # rounding does not depend on the block (elementwise), the moments are BIT-identical for
        # every width and task count — rows and columns alike, including a ragged last block.
        dim, N, R = 50, 12, 11
        d = collect(range(-0.9, 0.95; length = dim))
        rowop!(Y, X) = (Y .= X .* d'; nothing)
        colop!(Y, X) = (Y .= d .* X; nothing)
        run(E, template, f!, nt) =
            K.kpm_moments(f!, N, R, template, UniPolfed.CPU(); rng = MersenneTwister(3), ntasks = nt)

        for E in (Float64, ComplexF64)
            ref_row = run(E, zeros(E, 1, dim), rowop!, 1)
            ref_col = run(E, zeros(E, dim), colop!, 1)
            for nb in (1, 3, 4), nt in (1, 3)
                @test run(E, zeros(E, nb, dim), rowop!, nt) == ref_row
                @test run(E, nb == 1 ? zeros(E, dim) : zeros(E, dim, nb), colop!, nt) == ref_col
            end
            @test ref_row ≈ ref_col   # same probes; only the summation order differs
        end
    end

    @testset "ntasks is bit-identical with a generic sparse operator" begin
        Random.seed!(11)
        dim = 80
        A = sprandn(dim, dim, 0.1)
        H = (A + A') ./ (2 * opnorm(Matrix(A + A')))
        f!(Y, X) = mul!(Y, X, H)
        μ(nt) = UniPolfed.with_serial_libraries() do   # as calculate_dos runs it
            K.kpm_moments(f!, 20, 13, zeros(4, dim), UniPolfed.CPU();
                          rng = MersenneTwister(5), ntasks = nt)
        end
        @test μ(1) == μ(2) == μ(5)
        @test μ(1)[1] ≈ dim   # μ₀ = Tr I
    end

    @testset "kpm_moments and probe_moments! are type-stable" begin
        d = collect(range(-0.9, 0.9; length = 30))
        f!(Y, X) = (Y .= X .* d'; nothing)
        @inferred K.kpm_moments(f!, 8, 5, zeros(2, 30), UniPolfed.CPU(); ntasks = 2)
        @inferred K.kpm_moments(f!, 8, 5, zeros(ComplexF32, 3, 30), UniPolfed.CPU())
        P = Matrix{Float64}(I, 30, 30)[1:3, :]
        @inferred K.probe_moments!(zeros(8, 3), f!, P, K.ChebyshevWorkspace(P))
    end

    @testset "kpm_moments rejects a non-positive ntasks" begin
        @test_throws ArgumentError K.kpm_moments((y, x) -> (y .= x), 2, 2, zeros(4), UniPolfed.CPU();
                                                 ntasks = 0)
    end

    @testset "kpm_moments H = cI is exact" begin
        N = 8
        R = 5
        dim = 6
        c = 0.4

        template = zeros(Float64, dim)
        f!(y, x) = (y .= c .* x)

        μ = K.kpm_moments(f!, N, R, template, UniPolfed.CPU())

        expected = [dim * _kpm_test_cheb(c, k) for k in 0:N-1]
        @test μ ≈ expected
    end

    @testset "kpm_moments supports element type $E" for E in (Float32, Float64, ComplexF32, ComplexF64)
        S = real(E)
        N = 8
        R = 5
        dim = 6
        c = S(0.4)
        tol = sqrt(eps(S))

        template = zeros(E, dim)
        f!(y, x) = (y .= c .* x)

        μ = K.kpm_moments(f!, N, R, template, UniPolfed.CPU())

        expected = [dim * _kpm_test_cheb(c, k) for k in 0:N-1]
        @test eltype(μ) === S
        @test isapprox(μ, expected; rtol = tol, atol = tol)
    end

    @testset "kpm_moments counts matrix block entries toward R" begin
        N = 2
        R = 5
        dim = 6
        block = 2
        c = 0.4
        expected = [dim * _kpm_test_cheb(c, k) for k in 0:N-1]

        col_shapes = Set{Tuple{Int,Int}}()
        col_calls = Ref(0)
        f_col!(Y, X) = (col_calls[] += 1; push!(col_shapes, size(X)); Y .= c .* X)
        μ_col = K.kpm_moments(f_col!, N, R, zeros(Float64, dim, block), UniPolfed.CPU())

        row_shapes = Set{Tuple{Int,Int}}()
        row_calls = Ref(0)
        f_row!(Y, X) = (row_calls[] += 1; push!(row_shapes, size(X)); Y .= c .* X)
        μ_row = K.kpm_moments(f_row!, N, R, zeros(Float64, block, dim), UniPolfed.CPU())

        @test μ_col ≈ expected
        @test μ_row ≈ expected
        @test col_shapes == Set([(dim, block)])
        @test row_shapes == Set([(block, dim)])
        @test col_calls[] == cld(R, block)
        @test row_calls[] == cld(R, block)
    end

    @testset "row and column probes agree for symmetric H" begin
        H = Diagonal([0.3, -0.7, 0.5, -0.1])
        N = 8
        R = 20

        f_col!(Y, X) = mul!(Y, H, X)
        f_row!(Y, X) = mul!(Y, X, H)

        Random.seed!(123)
        μ_col = K.kpm_moments(f_col!, N, R, zeros(Float64, size(H, 1), 1), UniPolfed.CPU())

        Random.seed!(123)
        μ_row = K.kpm_moments(f_row!, N, R, zeros(Float64, 1, size(H, 1)), UniPolfed.CPU())

        @test μ_row ≈ μ_col
    end

    @testset "complex Hermitian row and column actions agree" begin
        H = ComplexF64[
            0.20       0.10 + 0.20im  0.05 - 0.03im
            0.10 - 0.20im  -0.30      0.07 + 0.04im
            0.05 + 0.03im   0.07 - 0.04im  0.40
        ]
        λ = eigvals(Hermitian(H))
        @test all(abs.(λ) .< 1)

        N = 9
        dim = size(H, 1)
        expected = [sum(_kpm_test_cheb(λj, k) for λj in λ) for k in 0:N-1]

        probe_col = Matrix{ComplexF64}(I, dim, dim)
        μ_col = zeros(Float64, N)
        f_col!(Y, X) = mul!(Y, H, X)
        K.trace_doubling!(f_col!, probe_col, μ_col, K.ChebyshevWorkspace(probe_col))

        probe_row = Matrix{ComplexF64}(I, dim, dim)
        μ_row = zeros(Float64, N)
        # Row probes store the transpose of the column-ket representation, so the
        # equivalent right action is X * transpose(H). For real symmetric H this is X * H.
        f_row!(Y, X) = mul!(Y, X, transpose(H))
        K.trace_doubling!(f_row!, probe_row, μ_row, K.ChebyshevWorkspace(probe_row))

        @test μ_col ≈ expected atol=1e-12 rtol=1e-12
        @test μ_row ≈ expected atol=1e-12 rtol=1e-12
        @test μ_row ≈ μ_col atol=1e-12 rtol=1e-12
    end


end
