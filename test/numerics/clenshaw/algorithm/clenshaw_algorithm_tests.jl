using Test
using Polfed
using LinearAlgebra

CL = Polfed.Numerics.Clenshaw

# Ground-truth helpers (independent of the implementation under test).
single_term(n) = k -> (k == n ? 1.0 : 0.0)          # expansion = φ_n alone
function cheb_matrix(A, n)                            # T_n(A) via matrix recurrence
    n == 0 && return Matrix{eltype(A)}(I, size(A)...)
    Tprev, Tcur = Matrix{eltype(A)}(I, size(A)...), copy(A)
    for _ in 2:n
        Tprev, Tcur = Tcur, 2A * Tcur - Tprev
    end
    return Tcur
end

@testset "clenshaw_algorithm!" begin

    @testset "scalar loop: Chebyshev Tₙ(cosθ)=cos(nθ)" begin
        α, β, ϕ₀, ϕ₁ = CL.polynomial_properties[:Chebyshev]
        θ = 0.37
        for n in (0, 1, 2, 5, 10, 100)
            b = zeros(Float64, 3)
            y = CL.clenshaw_algorithm!(single_term(n), n, α, β, ϕ₀, ϕ₁, b, cos(θ))
            @test y ≈ cos(n * θ)
        end
    end

    @testset "scalar loop: Legendre low orders" begin
        α, β, ϕ₀, ϕ₁ = CL.polynomial_properties[:Legendre]
        x = 0.7
        truth = [1.0, x, (3x^2 - 1) / 2, (5x^3 - 3x) / 2]
        for n in 0:3
            b = zeros(Float64, 3)
            y = CL.clenshaw_algorithm!(single_term(n), n, α, β, ϕ₀, ϕ₁, b, x)
            @test y ≈ truth[n+1]
        end
    end

    @testset "scalar loop: ComplexF64 evaluation point" begin
        α, β, ϕ₀, ϕ₁ = CL.polynomial_properties[:Chebyshev]
        z = 0.5 + 0.2im
        b = zeros(ComplexF64, 3)
        y = CL.clenshaw_algorithm!(single_term(2), 2, α, β, ϕ₀, ϕ₁, b, z)
        @test y ≈ 2z^2 - 1
    end

    @testset "matrix loop: Tₙ(A)·v via build_clenshaw_kernels" begin
        A = [0.2 0.1; 0.1 -0.3]
        mapping!(out, in) = mul!(out, A, in)
        recurrence!, finalsum! = CL.build_clenshaw_kernels(mapping!, :Chebyshev)
        v = [1.0, -2.0]
        for n in (0, 1, 2, 3, 7)
            Y = zeros(2)
            b = [zeros(2) for _ in 1:3]
            CL.clenshaw_algorithm!(single_term(n), n, recurrence!, finalsum!, b, v, Y)
            @test Y ≈ cheb_matrix(A, n) * v
        end
    end

    @testset "edge: order 0 returns the constant term only" begin
        α, β, ϕ₀, ϕ₁ = CL.polynomial_properties[:Chebyshev]
        b = zeros(Float64, 3)
        y = CL.clenshaw_algorithm!(k -> 3.0, 0, α, β, ϕ₀, ϕ₁, b, 0.42)
        @test y ≈ 3.0                                  # c₀·ϕ₀, no loop iterations
    end
end
