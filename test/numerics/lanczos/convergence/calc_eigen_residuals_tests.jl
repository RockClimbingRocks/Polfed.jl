using Test
using LinearAlgebra
using Random
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

@testset "build_values_and_residuals (fused pass)" begin
    pu = CPU()

    @testset "agrees with the two-pass helpers" begin
        A = Symmetric(diagm([1.0, 2.5, -3.0, 4.0]))
        valsmap!(y, x) = mul!(y, A, x)
        vecs = Matrix(qr(randn(4, 3)).Q)                     # not eigenvectors: residuals ≠ 0

        vals_ref = LZ.build_eigenvalues(LZ.ColumnLayout(), vecs, valsmap!, pu)
        res_ref  = LZ.build_residuals(LZ.ColumnLayout(), vecs, vals_ref, valsmap!, pu)
        vals, res = LZ.build_values_and_residuals(LZ.ColumnLayout(), vecs, valsmap!, pu)

        @test vals ≈ vals_ref
        @test res  ≈ res_ref
        @test any(res .> 1e-3)                               # the check has teeth
    end

    @testset "applies the operator once per eigenvector" begin
        A = Symmetric(diagm([1.0, 2.0, 3.0]))
        applications = Ref(0)
        counting!(y, x) = (applications[] += 1; mul!(y, A, x))
        vecs = Matrix{Float64}(I, 3, 3)

        LZ.build_values_and_residuals(LZ.ColumnLayout(), vecs, counting!, pu)
        @test applications[] == 3                            # not 6 (two passes)
    end

    @testset "row layout and empty input" begin
        A = Symmetric(diagm([2.0, 5.0]))
        rowmap!(Y, X) = mul!(Y, X, A)
        vals, res = LZ.build_values_and_residuals(LZ.RowLayout(), Matrix{Float64}(I, 2, 2), rowmap!, pu)
        @test vals ≈ [2.0, 5.0]
        @test all(res .< 1e-12)

        vals0, res0 = LZ.build_values_and_residuals(LZ.ColumnLayout(), zeros(3, 0), rowmap!, pu)
        @test isempty(vals0) && isempty(res0)
    end
end

# ── memory-bounded extraction: the Krylov basis is the only full-size array ─────────────────

# A few Lanczos steps on `A` in `layout`; `f!` acts per layout (rows: Y = X·A).
function _eig_test_factorization(A, x0, layout; basistype = MatrixBasis, nsteps = 4)
    seed = LZ._prepare_seed(layout, x0)
    f! = layout isa RowLayout ? ((Y, X) -> mul!(Y, X, A)) : ((Y, X) -> mul!(Y, A, X))
    iterator = LZ.LanczosIterator(f!, seed, FullRO(), layout)
    fac = LZ.lanczositer(iterator, LZ.createbasis(40, seed, basistype, CPU(), layout), 40, CPU())
    for it in 2:(nsteps + 1)
        LZ.lanczositer!(iterator, fac, it) || break
    end
    return fac, f!
end

_eig_seed(layout, T, n, bs) = layout isa RowLayout ? randn(T, bs, n) : randn(T, n, bs)

const _EIG_CASES = ((ColumnLayout(), Float64), (RowLayout(), Float64),
                    (ColumnLayout(), ComplexF64), (RowLayout(), ComplexF64),
                    (RowLayout(), Float32))

@testset "calc_eigen_residuals_chunked matches the full reconstruction" begin
    Random.seed!(11)
    n = 40
    idxs = [2, 7, 1, 9, 4]
    @testset "$(nameof(typeof(layout))) $T" for (layout, T) in _EIG_CASES
        A = Matrix(Hermitian(randn(T, n, n)))
        fac, f! = _eig_test_factorization(A, _eig_seed(layout, T, n, 3), layout)
        _, ϕ, _ = LZ.calc_ritz_residuals(fac)
        vals_ref, _, res_ref = LZ.calc_eigen_residuals(fac, ϕ, idxs, f!)
        basis_before = copy(LZ.allvecs(fac.basis))
        for chunk in (1, 2, 5, 64)
            vals, res = LZ.calc_eigen_residuals_chunked(fac, ϕ, idxs, f!; chunk = chunk)
            @test vals isa Vector{real(T)}
            @test vals ≈ vals_ref
            @test res ≈ res_ref rtol = sqrt(eps(real(T)))
        end
        @test LZ.allvecs(fac.basis) == basis_before          # the basis survives a check
        @test all(isempty, LZ.calc_eigen_residuals_chunked(fac, ϕ, Int[], f!))
    end

    @testset "vector basis" begin
        A = Matrix(Symmetric(randn(n, n)))
        fac, f! = _eig_test_factorization(A, randn(n), ColumnLayout(); basistype = VectorBasis)
        _, ϕ, _ = LZ.calc_ritz_residuals(fac)
        vals_ref, _, res_ref = LZ.calc_eigen_residuals(fac, ϕ, [3, 1], f!)
        vals, res = LZ.calc_eigen_residuals_chunked(fac, ϕ, [3, 1], f!; chunk = 1)
        @test vals ≈ vals_ref && res ≈ res_ref
    end
end

@testset "extract_eigenpairs! builds the eigenvectors inside the basis" begin
    Random.seed!(12)
    n = 50
    idxs = [3, 1, 8, 5]
    @testset "$(nameof(typeof(layout))) $T, block = $block" for (layout, T) in _EIG_CASES,
                                                               block in (0, 1, 3, 7)
        A = Matrix(Hermitian(randn(T, n, n)))
        fac, f! = _eig_test_factorization(A, _eig_seed(layout, T, n, 3), layout)
        _, ϕ, _ = LZ.calc_ritz_residuals(fac)
        vals_ref, vecs_ref, res_ref = LZ.calc_eigen_residuals(fac, ϕ, idxs, f!)

        vals, vecs, res = LZ.extract_eigenpairs!(fac, ϕ, idxs, f!; block = block)
        @test parent(vecs) === fac.basis.basis                 # no eigenvector matrix allocated
        @test size(vecs) == size(vecs_ref)
        @test vecs ≈ vecs_ref
        @test vals ≈ vals_ref
        @test res ≈ res_ref rtol = sqrt(eps(real(T)))
    end

    @testset "empty selection does not pin the basis" begin
        fac, f! = _eig_test_factorization(Matrix(Symmetric(randn(n, n))), randn(3, n), RowLayout())
        _, ϕ, _ = LZ.calc_ritz_residuals(fac)
        vals, vecs, res = LZ.extract_eigenpairs!(fac, ϕ, Int[], f!)
        @test size(vecs) == (0, n) && isempty(vals) && isempty(res)
        @test parent(vecs) !== fac.basis.basis
    end

    @testset "vector basis falls back to a fresh matrix" begin
        A = Matrix(Symmetric(randn(n, n)))
        fac, f! = _eig_test_factorization(A, randn(n, 2), ColumnLayout(); basistype = VectorBasis)
        _, ϕ, _ = LZ.calc_ritz_residuals(fac)
        vals_ref, vecs_ref, _ = LZ.calc_eigen_residuals(fac, ϕ, idxs, f!)
        vals, vecs, _ = LZ.extract_eigenpairs!(fac, ϕ, idxs, f!)
        @test vecs isa Matrix && vecs ≈ vecs_ref && vals ≈ vals_ref
    end
end

@testset "permute_eigenvectors!" begin
    Random.seed!(13)
    n, D = 6, 11
    @testset "$(nameof(typeof(layout))), block = $block" for layout in (ColumnLayout(), RowLayout()),
                                                             block in (0, 1, 2, 5)
        # A leading view with spare slots behind it — the shape extract_eigenpairs! returns.
        M = layout isa RowLayout ? randn(n + 3, D) : randn(D, n + 3)
        spare = copy(layout isa RowLayout ? M[n+1:end, :] : M[:, n+1:end])
        vecs = LZ._leading(layout, M, n)
        ref = copy(vecs)
        perm = randperm(n)
        @test LZ.permute_eigenvectors!(layout, vecs, perm; block = block) === vecs
        @test vecs == (layout isa RowLayout ? ref[perm, :] : ref[:, perm])
        @test (layout isa RowLayout ? M[n+1:end, :] : M[:, n+1:end]) == spare

        plain = copy(ref)                                    # a plain matrix works too
        LZ.permute_eigenvectors!(layout, plain, perm; block = block)
        @test plain == vecs
    end

    @testset "identity is a no-op; misuse throws" begin
        M = randn(4, 9)
        ref = copy(M)
        LZ.permute_eigenvectors!(RowLayout(), M, 1:4)
        @test M == ref
        @test_throws DimensionMismatch LZ.permute_eigenvectors!(RowLayout(), M, [2, 1])
        @test_throws ArgumentError LZ.permute_eigenvectors!(RowLayout(), view(M, 2:3, :), [2, 1])
        @test_throws ArgumentError LZ.permute_eigenvectors!(ColumnLayout(), view(M, 1:2, 1:2), [2, 1])
    end
end

@testset "final extraction and sort allocate only scratch" begin
    # Row layout at a size where one eigenvector matrix is several MB: the extraction and the
    # sort must stay far below it (the old path allocated two such matrices).
    Random.seed!(14)
    D, n = 20_000, 30
    d = randn(D)
    rowdiag!(Y, X) = (Y .= X .* transpose(d))                # X·A for A = Diagonal(d)
    function consumed_factorization()
        seed = LZ._prepare_seed(RowLayout(), randn(4, D))
        it = LZ.LanczosIterator(rowdiag!, seed, FullRO(), RowLayout())
        fac = LZ.lanczositer(it, LZ.createbasis(48, seed, MatrixBasis, CPU(), RowLayout()), 48, CPU())
        for i in 2:11
            LZ.lanczositer!(it, fac, i)
        end
        _, ϕ, _ = LZ.calc_ritz_residuals(fac)
        return fac, ϕ
    end
    extract_and_sort(fac, ϕ) = begin
        vals, vecs, _ = LZ.extract_eigenpairs!(fac, ϕ, 1:n, rowdiag!; block = 256)
        LZ.permute_eigenvectors!(RowLayout(), vecs, sortperm(vals); block = 256)
    end
    extract_and_sort(consumed_factorization()...)            # compile
    fac, ϕ = consumed_factorization()
    one_matrix = n * D * sizeof(Float64)
    @test @allocated(extract_and_sort(fac, ϕ)) < one_matrix ÷ 4
end
