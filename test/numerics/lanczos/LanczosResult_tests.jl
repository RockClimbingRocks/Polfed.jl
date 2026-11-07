using Test
using Polfed
using Polfed.Numerics.Lanczos
using LinearAlgebra
using Random

Random.seed!(424242)

function _hc_sym(λ)
    n = length(λ)
    Q = Matrix(qr(randn(n, n)).Q)
    A = Q * Diagonal(λ) * Q'
    return Matrix(Hermitian((A + A') / 2))
end

@testset "has_converged (spectrum-gluing query)" begin
    λ = sort(vcat([-10.0, -8.0, -6.0, 6.0, 8.0, 10.0], range(-1.0, 1.0; length=34)))
    n = length(λ)
    A = _hc_sym(λ)
    exact = eigen(A).values
    x0 = Float64[1 / (i + 0.5) - 0.3cos(1.3i) + 0.2 for i in 1:n]   # deterministic generic seed

    r = lanczos(A, x0, LowestEigs(3); maxdim=n, eigentol=1e-10)

    @testset "converged eigenvalues are found" begin
        @test has_converged(r, exact[1]; atol=1e-10)
        @test has_converged(r, exact[2]; atol=1e-10)
        @test has_converged(r, exact[3]; atol=1e-10)
    end

    @testset "absent / out-of-window eigenvalues are not" begin
        @test !has_converged(r, 0.5; atol=1e-10)            # bulk value, not targeted
        @test !has_converged(r, exact[end]; atol=1e-10)     # the top end, not targeted here
    end

    @testset "atol gates distance, restol gates residual" begin
        @test  has_converged(r, r.vals[1] + 1e-11; atol=1e-10)   # 1e-11 offset < atol → found
        @test !has_converged(r, r.vals[1] + 1e-9;  atol=1e-10)   # 1e-9  offset > atol → not found
        @test  has_converged(r, exact[1]; atol=1e-10, restol=1e-6)   # residual is tiny
        @test !has_converged(r, exact[1]; atol=1e-10, restol=0.0)    # impossible residual bar
    end

    @testset "empty result returns false" begin
        empty_r = lanczos(A, x0, LowestEigs(1); maxdim=2)   # too few iters to converge
        if empty_r.converged == 0
            @test !has_converged(empty_r, exact[1]; atol=1e-10)
        end
    end
end
