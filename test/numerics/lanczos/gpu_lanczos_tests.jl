# Lanczos on device arrays: every factorization (scalar vector, block), layout (column, row) and
# basis (MatrixBasis, VectorBasis), real and complex, checked against LAPACK on the host. Gated on
# a working CUDA device (see `gpu_gate.jl`).
#
# What a device run must get right beyond the CPU path: the Krylov vectors stay on the device while
# the projected matrix, its Ritz vectors and every coupling block live on the host, so each step
# moves a `bs × bs` block across and back — and the row layout orthonormalizes without a device LQ.

using LinearAlgebra, SparseArrays, Random, Test
using Polfed: lanczos, lanczos_extrema, LowestEigs, MatrixBasis, VectorBasis, ColumnLayout, RowLayout

isdefined(Main, :GPU_OK) || include(joinpath(@__DIR__, "..", "..", "gpu_gate.jl"))

# QREM-like fixture (asymmetric spectrum): diag(ε) + hx·Σσˣ, plus an antisymmetric imaginary part
# for a complex T, so Hᵀ ≠ H and a conjugation slip cannot pass unnoticed.
function _gpu_lanczos_fixture(::Type{T}; L = 8) where {T}
    Random.seed!(20260707)
    D = 2^L
    d = randn(D)
    I_ = Int[]; J = Int[]; V = T[]
    for i in 1:D
        push!(I_, i); push!(J, i); push!(V, d[i])
        for b in 0:L-1
            push!(I_, i); push!(J, xor(i - 1, 1 << b) + 1); push!(V, 0.7)
        end
    end
    H = sparse(I_, J, V, D, D)
    T <: Complex || return H
    Random.seed!(7)
    P = sprand(D, D, 2 / D)
    return H + im * 0.3 * (P - P')
end

if !GPU_OK
    gpu_tests_not_run("GPU Lanczos tests")
else
    @testset "GPU Lanczos, $T" for T in (Float64, ComplexF64)
        H = _gpu_lanczos_fixture(T)
        D = size(H, 1)
        exact = eigvals(Hermitian(Matrix(H)))
        Hg = CUDA.CUSPARSE.CuSparseMatrixCSR(H)
        col! = (Y, X) -> (mul!(Y, Hg, X); nothing)
        row! = (Y, X) -> (mul!(Y, X, Hg); nothing)       # Y = X·H: rows are LEFT eigenvectors

        @testset "$label" for (label, x0, layout, basis) in (
                ("vector, MatrixBasis",          CuVector(randn(T, D)),   ColumnLayout(), MatrixBasis),
                ("vector, VectorBasis",          CuVector(randn(T, D)),   ColumnLayout(), VectorBasis),
                ("column block bs=4, MatrixBasis", CuArray(randn(T, D, 4)), ColumnLayout(), MatrixBasis),
                ("column block bs=4, VectorBasis", CuArray(randn(T, D, 4)), ColumnLayout(), VectorBasis),
                ("row block bs=1, MatrixBasis",  CuArray(randn(T, 1, D)), RowLayout(),    MatrixBasis),
                ("row block bs=4, MatrixBasis",  CuArray(randn(T, 4, D)), RowLayout(),    MatrixBasis))
            res = lanczos(layout isa RowLayout ? row! : col!, x0, LowestEigs(5);
                          layout = layout, basistype = basis, maxdim = 200)
            @test res.vals isa Vector                         # eigenvalues come back on the host
            @test length(res.vals) >= 5
            @test maximum(abs.(res.vals[1:5] .- exact[1:5])) < 1e-8
            V = Array(res.vecs)
            for i in 1:5
                # a row u with u·H = λu is the column eigenvector conj(u)
                v = layout isa RowLayout ? conj(V[i, :]) : V[:, i]
                @test norm(H * v - res.vals[i] * v) < 1e-6
            end
        end

        @testset "lanczos_extrema on a device matrix" begin
            Emin, Emax = lanczos_extrema(Hg)
            @test Emin ≈ exact[1] atol = 1e-6
            @test Emax ≈ exact[end] atol = 1e-6
        end
    end
end
