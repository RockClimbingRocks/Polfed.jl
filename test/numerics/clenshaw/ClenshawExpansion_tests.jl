using Test
using UniPolfed
using LinearAlgebra
using SparseArrays

CL = UniPolfed.Numerics.Clenshaw
ClenshawExpansion = CL.ClenshawExpansion

# Ground truth: T_n(A) via the matrix Chebyshev recurrence.
function cheb_mat(A, n)
    n == 0 && return Matrix{eltype(A)}(I, size(A)...)
    Tprev, Tcur = Matrix{eltype(A)}(I, size(A)...), copy(A)
    for _ in 2:n
        Tprev, Tcur = Tcur, 2A * Tcur - Tprev
    end
    return Tcur
end

# Duck-typed matrix-free operator (LinearOperators-style): mul!, size, eltype.
struct DuckOp{T}
    M::Matrix{T}
end
LinearAlgebra.mul!(y, op::DuckOp, x) = mul!(y, op.M, x)
Base.size(op::DuckOp, d) = size(op.M, d)
Base.eltype(op::DuckOp{T}) where {T} = T

dummy!(out, in) = nothing                          # scalar path never invokes the mapping

# Ground truth for a full expansion S(A) = Σ cₖ Tₖ(A).
cheb_expansion_mat(A, coeffs) = sum(coeffs[k+1] * cheb_mat(A, k) for k in 0:length(coeffs)-1)

# Test operator with `transpose(A) == A`: real-symmetric, or complex-SYMMETRIC (not
# Hermitian) so that S(A)ᵀ = S(A) and the row/column results match under `permutedims`.
function sym_test_matrix(::Type{T}) where {T}
    R = real(T)
    M = R[ 0.20  0.10  0.00 -0.10;
           0.10 -0.30  0.05  0.00;
           0.00  0.05  0.15  0.10;
          -0.10  0.00  0.10 -0.20]
    T <: Complex || return Matrix{T}(M)
    S = R[ 0.00  0.03 -0.02  0.01;
           0.03  0.00  0.04 -0.01;
          -0.02  0.04  0.00  0.02;
           0.01 -0.01  0.02  0.00]      # symmetric, zero diagonal ⇒ transpose(M+iS)=M+iS
    return Complex.(M, S)
end

# Deterministic block input (nonzero imaginary part for complex types).
function block_input(::Type{T}, nrows::Int, ncols::Int) where {T}
    scale = nrows * ncols
    T <: Complex && return T[(i + 0.3j)/scale + im*(0.2i - 0.1j)/scale for i in 1:nrows, j in 1:ncols]
    return T[(i + 0.3j)/scale for i in 1:nrows, j in 1:ncols]
end

@testset "ClenshawExpansion" begin

    @testset "scalar callable — all families" begin
        x = 0.7
        cases = (
            (:Chebyshev, [1.0, 0.7, 2*0.7^2 - 1, 4*0.7^3 - 3*0.7]),
            (:Legendre,  [1.0, 0.7, (3*0.7^2 - 1)/2, (5*0.7^3 - 3*0.7)/2]),
            (:Hermite,   [1.0, 2*0.7, 4*0.7^2 - 2, 8*0.7^3 - 12*0.7]),
            (:Taylor,    [1.0, 0.7, 0.7^2, 0.7^3]),
        )
        for (fam, truth) in cases
            for n in 0:3
                coeffs = [i == n ? 1.0 : 0.0 for i in 0:n]
                ce = ClenshawExpansion(fam, coeffs, n, dummy!, 1, Float64)
                @test ce(x) ≈ truth[n+1]
            end
        end
    end

    @testset "scalar callable — high order & complex" begin
        θ = 0.31
        ce = ClenshawExpansion(:Chebyshev, [i == 100 ? 1.0 : 0.0 for i in 0:100], 100, dummy!, 1, Float64)
        @test ce(cos(θ)) ≈ cos(100*θ)

        cz = ClenshawExpansion(:Chebyshev, [0.0, 0.0, 1.0], 2, dummy!, 1, ComplexF64)
        z = 0.5 + 0.2im
        @test cz(z) ≈ 2z^2 - 1
    end

    @testset "constructor: mapping! callback ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = T[0.2 0.1; 0.1 -0.3]
        mapping!(out, in) = mul!(out, A, in)
        v = T[1, -2]
        for n in (0, 1, 2, 5)
            coeffs = [i == n ? one(T) : zero(T) for i in 0:n]
            ce = ClenshawExpansion(:Chebyshev, coeffs, n, mapping!, size(A, 1), eltype(A))
            Y = zeros(T, 2)
            b = [zeros(T, 2) for _ in 1:3]
            ce(Y, v, b)
            @test Y ≈ cheb_mat(A, n) * v
        end
    end

    @testset "constructor: AbstractMatrix (dense & sparse)" begin
        A = [0.2 0.1; 0.1 -0.3]
        v = [1.0, -2.0]
        n = 4
        coeffs = [i == n ? 1.0 : 0.0 for i in 0:n]
        for M in (A, sparse(A))
            ce = ClenshawExpansion(:Chebyshev, coeffs, n, M)
            @test ce.D == 2 && ce.T == Float64
            Y = zeros(2); b = [zeros(2) for _ in 1:3]
            ce(Y, v, b)
            @test Y ≈ cheb_mat(A, n) * v
        end
    end

    @testset "constructor: duck-typed mul! operator (LinearOperator-style)" begin
        A = [0.2 0.1; 0.1 -0.3]
        op = DuckOp(A)
        v = [1.0, -2.0]
        n = 3
        coeffs = [i == n ? 1.0 : 0.0 for i in 0:n]
        ce = ClenshawExpansion(:Chebyshev, coeffs, n, op)
        @test ce.D == 2 && ce.T == Float64
        Y = zeros(2); b = [zeros(2) for _ in 1:3]
        ce(Y, v, b)
        @test Y ≈ cheb_mat(A, n) * v
    end

    @testset "constructor: explicit recurrence!/finalsum! matches mapping! path" begin
        A = [0.2 0.1; 0.1 -0.3]
        mapping!(out, in) = mul!(out, A, in)
        n = 4
        coeffs = [i == n ? 1.0 : 0.0 for i in 0:n]
        v = [1.0, -2.0]

        ce1 = ClenshawExpansion(:Chebyshev, coeffs, n, mapping!, 2, Float64)
        rec!, fin! = CL.build_clenshaw_kernels(mapping!, :Chebyshev)
        ce2 = ClenshawExpansion(:Chebyshev, coeffs, n, rec!, fin!, 2, Float64)

        Y1 = zeros(2); Y2 = zeros(2)
        ce1(Y1, v, [zeros(2) for _ in 1:3])
        ce2(Y2, v, [zeros(2) for _ in 1:3])
        @test Y1 ≈ Y2 ≈ cheb_mat(A, n) * v
    end

    @testset "constructor: scalar-only (no operator)" begin
        # single-term Chebyshev T₂ via the operator-free constructor
        ce = ClenshawExpansion(:Chebyshev, [0.0, 0.0, 1.0], 2)
        @test ce.D == 0 && ce.T == Float64
        θ = 0.4
        @test ce(cos(θ)) ≈ cos(2θ)

        # coefficients supplied as a function; Legendre P₃
        ce2 = ClenshawExpansion(:Legendre, k -> float(k == 3), 3)
        x = 0.6
        @test ce2(x) ≈ (5x^3 - 3x) / 2

        # the vector/matrix call path must error clearly
        @test_throws ErrorException ce(zeros(2), [1.0, 1.0], [zeros(2) for _ in 1:3])

        # unknown family still rejected at construction
        @test_throws ArgumentError ClenshawExpansion(:Bogus, [1.0], 0)
    end

    @testset "matrix callable: block (multi-column) input" begin
        A = [0.2 0.1; 0.1 -0.3]
        n = 3
        coeffs = [i == n ? 1.0 : 0.0 for i in 0:n]
        ce = ClenshawExpansion(:Chebyshev, coeffs, n, A)
        X = [1.0 0.0; -2.0 1.0]
        Y = zeros(2, 2); b = [zeros(2, 2) for _ in 1:3]
        ce(Y, X, b)
        @test Y ≈ cheb_mat(A, n) * X
    end

    @testset "matrix callable: ROW layout (bs×D) correctness ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        # Row layout: the mapping right-multiplies, out = in·A, with in/out shaped bs×D.
        # Clenshaw then builds the polynomial of "right-multiply by A", so the result is
        # Y_row = X_row · S(A) for ANY A — ground truth independent of symmetry.
        A   = sym_test_matrix(T)
        D   = size(A, 1)
        bs  = 3
        map_row!(out, in) = mul!(out, in, A)
        rtol = T <: Union{Float32,ComplexF32} ? 1f-3 : 1e-11

        # single-term expansions S(A) = Tₙ(A)
        for n in (0, 1, 2, 5)
            coeffs = [i == n ? one(T) : zero(T) for i in 0:n]
            ce = ClenshawExpansion(:Chebyshev, coeffs, n, map_row!, D, T)
            Xr = block_input(T, bs, D)
            Yr = zeros(T, bs, D); b = [zeros(T, bs, D) for _ in 1:3]
            ce(Yr, Xr, b)
            @test isapprox(Yr, Xr * cheb_mat(A, n); rtol = rtol)
        end

        # general multi-term expansion
        coeffs = T[0.3, -0.2, 0.5, 0.1, -0.4]
        ce = ClenshawExpansion(:Chebyshev, coeffs, 4, map_row!, D, T)
        Xr = block_input(T, bs, D)
        Yr = zeros(T, bs, D); b = [zeros(T, bs, D) for _ in 1:3]
        ce(Yr, Xr, b)
        @test isapprox(Yr, Xr * cheb_expansion_mat(A, coeffs); rtol = rtol)
    end

    @testset "ROW (bs×D) equals COLUMN (D×bs) transposed ($T)" for T in (Float64, ComplexF64)
        # transpose(A) == A ⇒ S(A)ᵀ = S(A), so a row-layout solve on Xᵀ must equal the
        # column-layout solve on X, transposed back. This is the apples-to-apples check.
        A  = sym_test_matrix(T)
        D  = size(A, 1)
        bs = 3
        map_col!(out, in) = mul!(out, A, in)     # H·X  (column, D×bs)
        map_row!(out, in) = mul!(out, in, A)     # X·H  (row,    bs×D)
        coeffs = T[0.3, -0.2, 0.5, 0.1, -0.4]; n = 4

        ce_col = ClenshawExpansion(:Chebyshev, coeffs, n, map_col!, D, T)
        ce_row = ClenshawExpansion(:Chebyshev, coeffs, n, map_row!, D, T)

        Xc = block_input(T, D, bs)               # D×bs
        Xr = permutedims(Xc)                     # bs×D  (transpose, NOT adjoint)

        Yc = zeros(T, D, bs); bc = [zeros(T, D, bs) for _ in 1:3]
        Yr = zeros(T, bs, D); br = [zeros(T, bs, D) for _ in 1:3]
        ce_col(Yc, Xc, bc)
        ce_row(Yr, Xr, br)

        @test permutedims(Yr) ≈ Yc               # row result, transposed, equals column result
        @test Yc ≈ cheb_expansion_mat(A, coeffs) * Xc   # and both equal the ground truth
    end

    @testset "matrix callable: UniformScaling seed forms full S(A)" begin
        A = [0.2 0.1; 0.1 -0.3]
        n = 3
        coeffs = [i == n ? 1.0 : 0.0 for i in 0:n]
        ce = ClenshawExpansion(:Chebyshev, coeffs, n, A)
        Y = zeros(2, 2); b = [zeros(2, 2) for _ in 1:3]
        ce(Y, I, b)
        @test Y ≈ cheb_mat(A, n)
    end

    @testset "coefficients: Function vs Vector give identical results" begin
        A = [0.2 0.1; 0.1 -0.3]
        n = 5
        cvec = [0.3, -0.1, 0.7, 0.0, 1.2, -0.4]
        cfun = k -> cvec[k+1]
        v = [1.0, -2.0]

        ce_vec = ClenshawExpansion(:Chebyshev, cvec, n, A)
        ce_fun = ClenshawExpansion(:Chebyshev, cfun, n, A)
        Yv = zeros(2); Yf = zeros(2)
        ce_vec(Yv, v, [zeros(2) for _ in 1:3])
        ce_fun(Yf, v, [zeros(2) for _ in 1:3])
        @test Yv ≈ Yf
        @test ce_vec(0.4) ≈ ce_fun(0.4)
    end

    @testset "edge: zero coefficients ⇒ zero output" begin
        A = [0.2 0.1; 0.1 -0.3]
        ce = ClenshawExpansion(:Chebyshev, zeros(6), 5, A)
        Y = ones(2); b = [zeros(2) for _ in 1:3]
        ce(Y, [1.0, -2.0], b)
        @test all(iszero, Y)
        @test ce(0.7) ≈ 0.0
    end

    @testset "family accessor & scalar type stability" begin
        ce = ClenshawExpansion(:Chebyshev, [0.0, 0.0, 1.0], 2)
        @test CL.family(ce) == :Chebyshev
        ce_mat = ClenshawExpansion(:Legendre, [1.0, 0.0], 1, [1.0 0.0; 0.0 1.0])
        @test CL.family(ce_mat) == :Legendre

        # regression guard: the scalar path must infer a concrete return type
        # (family is a type parameter ⇒ recurrence weights resolve statically)
        @test @inferred(ce(0.4)) ≈ cos(2 * acos(0.4))
    end

    @testset "errors" begin
        mapping!(out, in) = mul!(out, [1.0 0; 0 1.0], in)
        # unknown polynomial family
        @test_throws ArgumentError ClenshawExpansion(:Bogus, [1.0], 0, mapping!, 2, Float64)
        # coefficient vector too short for the requested order
        @test_throws ArgumentError ClenshawExpansion(:Chebyshev, [1.0, 2.0], 5, mapping!, 2, Float64)
        # workspace vector `b` must have length 3
        ce = ClenshawExpansion(:Chebyshev, [1.0, 0.0, 1.0], 2, [1.0 0; 0 1.0])
        @test_throws AssertionError ce(zeros(2), [1.0, 1.0], [zeros(2), zeros(2)])
    end
end
