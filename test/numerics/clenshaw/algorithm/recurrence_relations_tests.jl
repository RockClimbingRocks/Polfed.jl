using Test
using UniPolfed
using LinearAlgebra

CL = UniPolfed.Numerics.Clenshaw

@testset "recurrence_relations" begin

    @testset "mapping_recurrence! — vector ($T)" for T in (Float32, Float64,  ComplexF32, ComplexF64)
        A = T[2 1; 0 3]
        mapping!(out, in) = mul!(out, A, in)
        b1 = zeros(T, 2)
        b2 = T[1, 2]
        b3 = T[0.5, -0.5]
        c, α, β = T(1.5), 2.0, -1.0
        X = T[1, 1]

        ret = CL.mapping_recurrence!(mapping!, b1, b2, b3, c, α, β, X)
        expected = c .* X .+ α .* (A * b2) .+ β .* b3

        @test b1 ≈ expected
        @test ret === b1                      # mutates and returns the same buffer
    end

    @testset "mapping_recurrence! — matrix" begin
        A = [2.0 1.0; 0.0 3.0]
        mapping!(out, in) = mul!(out, A, in)
        b1 = zeros(2, 2)
        b2 = [1.0 2.0; 3.0 4.0]
        b3 = [0.1 0.2; 0.3 0.4]
        c, α, β = 1.5, 2.0, -1.0
        X = [1.0 0.0; 0.0 1.0]

        CL.mapping_recurrence!(mapping!, b1, b2, b3, c, α, β, X)
        @test b1 ≈ c .* X .+ α .* (A * b2) .+ β .* b3
    end

    @testset "mapping_recurrence! — UniformScaling seed (matrix buffers)" begin
        A = [2.0 1.0; 0.0 3.0]
        mapping!(out, in) = mul!(out, A, in)
        b1 = zeros(2, 2)
        b2 = [1.0 2.0; 3.0 4.0]
        b3 = [0.1 0.2; 0.3 0.4]
        c, α, β = 1.5, 2.0, -1.0

        CL.mapping_recurrence!(mapping!, b1, b2, b3, c, α, β, I)
        @test b1 ≈ c .* Matrix(I, 2, 2) .+ α .* (A * b2) .+ β .* b3
    end

    @testset "scalar_recurrence ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        c, α, β = T(1.5), 2.0, -1.0
        x, b2, b3 = T(0.7), T(0.3), T(-0.2)
        @test CL.scalar_recurrence(c, α, β, x, b2, b3) ≈ c + α * x * b2 + β * b3
    end

    @testset "edge: zero state and identity mapping" begin
        idmap!(out, in) = copyto!(out, in)
        b1 = ones(3)
        b2 = zeros(3)
        b3 = zeros(3)
        CL.mapping_recurrence!(idmap!, b1, b2, b3, 0.0, 2.0, -1.0, zeros(3))
        @test all(iszero, b1)                  # all sources zero ⇒ result zero
    end
end
