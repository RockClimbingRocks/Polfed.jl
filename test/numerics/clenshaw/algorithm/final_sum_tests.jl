using Test
using UniPolfed
using LinearAlgebra

CL = UniPolfed.Numerics.Clenshaw

@testset "final_sum" begin

    @testset "mapping_finalsum! — vector ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = T[2 1; 0 3]
        mapping!(out, in) = mul!(out, A, in)
        b2 = T[1, 2]
        b3 = T[0.5, -0.5]
        Y = zeros(T, 2)
        c₀, β₁, ϕ₀, ϕ₁ = T(1.5), -1.0, 1.0, 1.0
        X = T[1, 1]

        ret = CL.mapping_finalsum!(mapping!, b2, b3, c₀, β₁, ϕ₀, ϕ₁, Y, X)
        expected = c₀ .* ϕ₀ .* X .+ ϕ₁ .* (A * b2) .+ β₁ .* ϕ₀ .* b3

        @test Y ≈ expected
        @test ret === Y
    end

    @testset "mapping_finalsum! — UniformScaling seed (matrix buffers)" begin
        A = [2.0 1.0; 0.0 3.0]
        mapping!(out, in) = mul!(out, A, in)
        b2 = [1.0 2.0; 3.0 4.0]
        b3 = [0.1 0.2; 0.3 0.4]
        Y = zeros(2, 2)
        c₀, β₁, ϕ₀, ϕ₁ = 1.5, -1.0, 1.0, 2.0

        CL.mapping_finalsum!(mapping!, b2, b3, c₀, β₁, ϕ₀, ϕ₁, Y, I)
        @test Y ≈ c₀ .* ϕ₀ .* Matrix(I, 2, 2) .+ ϕ₁ .* (A * b2) .+ β₁ .* ϕ₀ .* b3
    end

    @testset "scalar_finalsum ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        c₀, β₁, ϕ₀, ϕ₁ = T(1.5), -1.0, 1.0, 2.0
        x, b2, b3 = T(0.7), T(0.3), T(-0.2)
        @test CL.scalar_finalsum(c₀, β₁, ϕ₀, ϕ₁, x, b2, b3) ≈ c₀ * ϕ₀ + ϕ₁ * x * b2 + β₁ * ϕ₀ * b3
    end

    @testset "edge: zero coefficient ⇒ zero output" begin
        A = [2.0 1.0; 0.0 3.0]
        mapping!(out, in) = mul!(out, A, in)
        Y = ones(2)
        CL.mapping_finalsum!(mapping!, zeros(2), zeros(2), 0.0, -1.0, 1.0, 1.0, Y, zeros(2))
        @test all(iszero, Y)
    end
end
