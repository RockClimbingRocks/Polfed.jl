using LinearAlgebra
using Random
using Test

using Polfed

const CUDA_TEST_AVAILABLE = let
    try
        @eval import CUDA
        Polfed.CUDA_AVAILABLE && CUDA.functional()
    catch
        false
    end
end

function check_ordered_eigenpairs(mat::AbstractMatrix, vals::AbstractVector, vecs::AbstractMatrix; atol=1e-6)
    vals_cpu = Vector(vals)
    vecs_cpu = Matrix(vecs)

    @test issorted(vals_cpu)

    for j in eachindex(vals_cpu)
        v = view(vecs_cpu, :, j)
        residual = norm(mat * v - vals_cpu[j] * v)
        @test residual <= atol * max(1, abs(vals_cpu[j]))
    end
end

function small_polfed_problem(; use_gpu::Bool=false)
    mat_cpu = Matrix(Diagonal([-3.0, -1.25, -0.2, 0.55, 1.8, 3.2]))
    blocksize = 2
    howmany = 3

    rng = MersenneTwister(1234)
    q = Matrix(qr(randn(rng, size(mat_cpu, 1), blocksize)).Q)[:, 1:blocksize]

    mat = use_gpu ? CUDA.CuArray(mat_cpu) : mat_cpu
    x0 = use_gpu ? CUDA.CuArray(q) : q

    mapping = Polfed.MappingConfig(
        parallel_strategy=Polfed.NoParallel(),
        Emin=minimum(diag(mat_cpu)),
        Emax=maximum(diag(mat_cpu)),
    )
    transform = Polfed.TransformConfig(order=12)
    fact = Polfed.FactorizationConfig(tol=1e-11, eigentol=1e-7, overestimate_iters=1.0)
    dos = Polfed.DoSConfig(N=8, R=2)

    vals, vecs = Polfed.polfed(
        mat,
        x0,
        howmany,
        0.0;
        mapping=mapping,
        transform=transform,
        fact=fact,
        dos=dos,
    )

    return mat_cpu, vals, vecs
end

@testset "in-place column permutation" begin
    A0 = reshape(collect(1:20), 5, 4)
    p = [3, 1, 4, 2]

    A = copy(A0)
    returned = Polfed.Lanczos.permute_columns_inplace!(A, p, length(p))

    @test returned === A
    @test A == A0[:, p]
    @test A isa Matrix
    @test !(A isa Base.SubArray)

    B0 = reshape(collect(1:30), 5, 6)
    q = [1, 5, 3, 6, 2, 4]

    B = copy(B0)
    Polfed.Lanczos.permute_columns_inplace!(B, q, length(q))

    @test B == B0[:, q]

    C0 = reshape(collect(1:30), 5, 6)
    r = [4, 1, 3, 2]
    expected = copy(C0)
    expected[:, 1:length(r)] .= C0[:, r]

    C = copy(C0)
    Polfed.Lanczos.permute_columns_inplace!(C, r, length(r))

    @test C == expected
end

@testset "small CPU polfed ordering" begin
    mat, vals, vecs = small_polfed_problem()

    @test vals isa Vector
    @test vecs isa Matrix
    @test !(vals isa Base.SubArray)
    @test !(vecs isa Base.SubArray)
    @test size(vecs, 2) == length(vals)

    check_ordered_eigenpairs(mat, vals, vecs)
end

if CUDA_TEST_AVAILABLE
    @testset "small GPU polfed ordering" begin
        A0 = reshape(collect(Float64, 1:20), 5, 4)
        p = [3, 1, 4, 2]

        A = CUDA.CuArray(A0)
        returned = Polfed.Lanczos.permute_columns_inplace!(A, p, length(p))

        @test returned === A
        @test Array(A) == A0[:, p]
        @test typeof(A) <: CUDA.CuArray{<:Any,2}
        @test !(A isa Base.SubArray)

        mat, vals, vecs = small_polfed_problem(use_gpu=true)

        @test typeof(vals) <: CUDA.CuArray{<:Any,1}
        @test typeof(vecs) <: CUDA.CuArray{<:Any,2}
        @test !(vals isa Base.SubArray)
        @test !(vecs isa Base.SubArray)
        @test size(vecs, 2) == length(vals)

        check_ordered_eigenpairs(mat, vals, vecs; atol=1e-5)
    end
end
