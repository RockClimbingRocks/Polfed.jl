using Test
using UniPolfed
using LinearAlgebra
using Random

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
