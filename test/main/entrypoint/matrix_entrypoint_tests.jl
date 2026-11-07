# End-to-end tests for the matrix entrypoint `polfed(H, x0, howmany, target)` (Stage C/D).
#
# The matrix entrypoint owns H, so it estimates the spectral bounds, picks the block layout (row by
# default on the CPU), builds that layout's operators (value-grouped row kernels or a generic mul!
# fallback), orients x0 for it, and delegates to the pipeline. Ground truth here is EXACT
# diagonalization: for a structured (QREM) Hamiltonian and an interior target, the eigenvalues
# POLFED returns must be the true eigenvalues of H nearest the target, to machine precision — for
# every optimize_mapping choice (generic + each value-grouped kernel) and both single-vector and
# block seeds.

using LinearAlgebra, SparseArrays, Random
using Polfed: polfed, MappingConfig, DoSConfig, PolfedResult

# QREM: diag(ε) + hx·Σᵢ σᵢˣ — single off-diagonal value, L single-bit-flip neighbours per row
# (the structured regime the row kernels target). Real-symmetric, so a genuine Hermitian operator.
function _qrem(L::Int; hx = 0.7, T = Float64)
    D = 2^L
    diagv = randn(T, D)
    I = Int[]; J = Int[]; V = T[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, T(hx))
        end
    end
    return sparse(I, J, V, D, D)
end

# The `k` values in `vals` closest to `E`, sorted — for comparing against the exact spectrum.
_closest(vals, E, k) = sort(sort(collect(vals); by = λ -> abs(λ - E))[1:k])

@testset "matrix entrypoint polfed(H, …) — L = $L" for L in (8, 10)
    Random.seed!(20260707)
    D = 2^L
    H = _qrem(L)
    exact = sort(eigvals(Symmetric(Matrix(H))))
    target_E = exact[cld(D, 2)] + 0.13     # an interior, non-degenerate target energy
    howmany = 6

    # `polfed(H, …)` returns a PolfedResult whose (interval-filtered) eigenvalues are the true
    # eigenvalues of H nearest the target.
    function check(res)
        @test res isa PolfedResult
        vals = sort(collect(res.vals))
        @test res.converged >= howmany
        @test !isempty(vals)
        # every returned value is a genuine eigenvalue of H
        for λ in vals
            @test minimum(abs.(exact .- λ)) < 1e-6
        end
        # the nearest returned eigenvalues match exact diagonalization
        k = min(length(vals), howmany)
        @test maximum(abs.(_closest(vals, target_E, k) .- _closest(exact, target_E, k))) < 1e-6
    end

    @testset "optimize_mapping = $(repr(choice)) (vector seed, bs=1)" for choice in
            (false, true, :auto, :universal, MappingKernel(:universal, :row_bucketed))
        x0 = randn(D)
        res = polfed(H, x0, howmany, target_E; mc = MappingConfig(optimize_mapping = choice))
        check(res)
    end

    @testset "block seed bs=$bs, optimize_mapping=$(repr(choice))" for (bs, choice) in
            ((2, :universal), (4, MappingKernel(:universal, :row_read)), (4, false))
        x0 = randn(D, bs)
        res = polfed(H, x0, howmany, target_E; mc = MappingConfig(optimize_mapping = choice))
        check(res)
    end

    @testset "NamedTuple pins kernel + params" begin
        x0 = randn(D, 2)
        res = polfed(H, x0, howmany, target_E;
                     mc = MappingConfig(optimize_mapping = (kernel = :universal,)))
        check(res)
    end

    @testset "explicit Emin/Emax skips the bounds probe" begin
        Emin, Emax = first(exact) - 0.5, last(exact) + 0.5
        res = polfed(H, randn(D), howmany, target_E;
                     mc = MappingConfig(optimize_mapping = :universal, Emin = Emin, Emax = Emax))
        check(res)
    end

    @testset "non-square matrix is rejected" begin
        @test_throws DimensionMismatch polfed(sprand(8, 9, 0.3), randn(8), 2, 0.0)
    end
end

@testset "matrix entrypoint prepares the DoS for an operator it owns" begin
    Random.seed!(3)
    H = _qrem(6)
    before = BLAS.get_num_threads()
    _, _, _, dosc, _ = Polfed._prepare_matrix_pipeline(H, randn(size(H, 1), 2), MappingConfig(),
                                                         DoSConfig(R = 40))
    @test dosc.R == 40
    @test dosc.block_size == Polfed.Defaults.polfed.dos_block_size
    @test dosc.tasks == Threads.nthreads()
    @test BLAS.get_num_threads() == before    # the serial bounds probe restored it

    _, _, _, kept, _ = Polfed._prepare_matrix_pipeline(H, randn(size(H, 1)), MappingConfig(),
                                                         DoSConfig(block_size = 2, tasks = 1))
    @test (kept.block_size, kept.tasks) == (2, 1)
end

@testset "matrix entrypoint layout: resolution and seed orientation" begin
    @test Polfed._matrix_layout(nothing, Polfed.CPU(), 1) == RowLayout()
    @test Polfed._matrix_layout(nothing, Polfed.CPU(), 16) == RowLayout()
    @test Polfed._matrix_layout(ColumnLayout(), Polfed.CPU(), 4) == ColumnLayout()
    @test Polfed._matrix_layout(RowLayout(), Polfed.CPU(), 1) == RowLayout()

    D = 12
    v, C, R = randn(D), randn(D, 3), randn(3, D)
    @test size(Polfed._orient_seed(RowLayout(), v, false)) == (1, D)
    @test Polfed._orient_seed(RowLayout(), C, false) == permutedims(C)
    @test Polfed._orient_seed(RowLayout(), R, true) === R
    @test Polfed._orient_seed(ColumnLayout(), v, false) === v      # stays a vector: scalar Lanczos
    @test Polfed._orient_seed(ColumnLayout(), C, false) === C
    @test Polfed._orient_seed(ColumnLayout(), R, true) == permutedims(R)
end

# Eigenvectors are checked with `collect`: a row-layout eigenvector is a strided row view, and
# MKLSparse 3.0's `mul!(y, A::SparseMatrixCSC, x::StridedVector)` ignores the stride of `x`.
@testset "matrix entrypoint in both layouts, $T" for T in (Float64, ComplexF64)
    Random.seed!(20260707)
    L = 8; D = 2^L
    H = _qrem(L)
    if T <: Complex                # a genuinely complex Hermitian H (Hᵀ ≠ H)
        Random.seed!(7)
        P = sprand(D, D, 2 / D)
        H = H + im * 0.3 * (P - P')
    end
    exact = sort(eigvals(Hermitian(Matrix(H))))
    target_E = exact[cld(D, 2)] + 0.13
    howmany = 6
    @testset "$(nameof(typeof(layout))), bs = $bs" for layout in (RowLayout(), ColumnLayout()),
                                                      bs in (1, 3)
        x0 = bs == 1 ? randn(T, D) : randn(T, D, bs)
        res = polfed(H, x0, howmany, target_E; mc = MappingConfig(layout = layout))
        vals = collect(res.vals)
        @test maximum(abs.(_closest(vals, target_E, howmany) .- _closest(exact, target_E, howmany))) < 1e-6
        # the returned vectors are eigenvectors of H itself, not of conj(H)
        for i in eachindex(vals)
            v = collect(eigenvector(res, i))
            @test norm(H * v - vals[i] * v) < 1e-6
        end
    end
    @testset "value-grouped kernels need the row layout" begin
        @test_throws ArgumentError polfed(H, randn(T, D), howmany, target_E;
                                          mc = MappingConfig(layout = ColumnLayout(),
                                                             optimize_mapping = :universal))
    end
end
