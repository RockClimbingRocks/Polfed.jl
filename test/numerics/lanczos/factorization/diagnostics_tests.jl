using Test
using LinearAlgebra
using Polfed
using Polfed.Numerics.Lanczos

LZ = Polfed.Numerics.Lanczos

@testset "diagnostics" begin

    @testset "_isfinite_number" begin
        @test LZ._isfinite_number(1.0)
        @test !LZ._isfinite_number(NaN)
        @test !LZ._isfinite_number(Inf)
        @test LZ._isfinite_number(1.0 + 2.0im)
        @test !LZ._isfinite_number(1.0 + NaN*im)     # non-finite imaginary part
        @test !LZ._isfinite_number(Inf + 0.0im)
    end

    @testset "has_nonfinite" begin
        @test !LZ.has_nonfinite([1.0, 2.0, 3.0])
        @test LZ.has_nonfinite([1.0, NaN, 3.0])
        @test LZ.has_nonfinite([1.0, Inf])
        @test !LZ.has_nonfinite(ComplexF64[1+1im, 2+2im])
        @test LZ.has_nonfinite(ComplexF64[1+1im, NaN+0im])
        @test !LZ.has_nonfinite(Float64[])            # empty → no non-finite values
    end

    @testset "krylov_symmetry_defect — scalar Lanczos is exactly 0" begin
        # The scalar projected matrix is SymTridiagonal by construction, so defect ≡ 0.
        A = Symmetric([2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0])
        f!(y, x) = mul!(y, A, x)
        x0 = normalize([1.0, 0.5, -0.3])
        iter = LZ.LanczosIterator(f!, x0, LZ.FullRO())
        basis = LZ.createbasis(20, x0, MatrixBasis, CPU())
        fac = LZ.lanczositer(iter, basis, 20, CPU())
        @test LZ.krylov_symmetry_defect(fac) == 0.0
    end

    @testset "krylov_symmetry_defect — block: symmetric mat ≈ 0, asymmetric > 0" begin
        n = 6
        A = Symmetric([1.0/(i+j) + (i==j) for i in 1:n, j in 1:n])
        f!(y, x) = mul!(y, A, x)
        x0 = Matrix(qr(Float64[i==j ? 1.0 : 0.1 for i in 1:n, j in 1:2]).Q)  # 6×2 orthonormal
        iter = LZ.LanczosIterator(f!, x0, LZ.FullRO())
        basis = LZ.createbasis(40, x0, MatrixBasis, CPU())
        fac = LZ.lanczositer(iter, basis, 40, CPU())

        # After init the projected block (just α₁) is Hermitian → defect ≈ 0.
        @test LZ.krylov_symmetry_defect(fac) < sqrt(eps(Float64))

        # Manually break symmetry of the stored projected matrix → defect > 0.
        fac.mat[2, 1] += 1.0
        @test LZ.krylov_symmetry_defect(fac) > sqrt(eps(Float64))
    end
end
