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

@testset "lanczos extrema helper" begin
    mat = Matrix(Diagonal([-3.0, -1.25, -0.2, 0.55, 1.8, 3.2]))
    x0 = ones(size(mat, 1))

    seed = Polfed.Lanczos._normalize_extrema_seed(x0)

    @test seed !== x0
    @test norm(seed) ≈ 1.0
    @test x0 == ones(size(mat, 1))

    @test_throws ArgumentError Polfed.Lanczos._normalize_extrema_seed(zeros(3))
    @test_throws DimensionMismatch Polfed.lanczos_extrema(ones(2, 3); x0=ones(2))
    @test_throws DimensionMismatch Polfed.lanczos_extrema(mat; x0=ones(size(mat, 1) - 1))
end

@testset "target specification normalization" begin
    core = Polfed.PolfedCore

    @test core.normalize_target_spec(:maxdos) isa core.TargetMaxDoS
    @test core.normalize_target_spec(:middle) isa core.TargetMiddle

    offset = core.normalize_target_spec((:offset, 0.25))
    @test offset isa core.TargetOffset
    @test offset.frac == 0.25

    absolute = core.normalize_target_spec((:unrescaled, -1.2))
    @test absolute isa core.TargetAbsolute
    @test absolute.value == -1.2
    @test core.normalize_target_spec(0.7) isa core.TargetAbsolute

    rescaled = core.normalize_target_spec((:rescaled, 0.4))
    @test rescaled isa core.TargetRescaled
    @test rescaled.value == 0.4
    @test core.normalize_target_spec(rescaled) === rescaled

    @test_throws ArgumentError core.normalize_target_spec(:unknown)
    @test_throws ArgumentError core.normalize_target_spec((:offset, 1.5))
    @test_throws ArgumentError core.normalize_target_spec((:offset, "0.2"))
    @test_throws ArgumentError core.normalize_target_spec((:absolute, 0.0))
    @test_throws ArgumentError core.normalize_target_spec((:quantile, 0.5))
    @test_throws ArgumentError core.normalize_target_spec((:other, 0.0))
    @test_throws ArgumentError core.normalize_target_spec("middle")
end

@testset "polfed initial state checks" begin
    @test Polfed.PolfedCore._initial_state_tolerance(rand(2)) == 1e-10
    @test Polfed.PolfedCore._initial_state_tolerance(rand(Float32, 2)) == 10eps(Float32)

    v0 = [3.0, 4.0]
    v = copy(v0)

    v_prepared = @test_logs (:warn, r"initial vector is not normalized") Polfed.PolfedCore._prepare_polfed_initial_state(v)

    @test v == v0
    @test norm(v_prepared) ≈ 1.0

    x0 = [
        1.0  1.0
        0.0  1.0
        1.0  0.0
    ]
    x = copy(x0)

    x_prepared = @test_logs (:warn, r"initial matrix is not orthonormal") Polfed.PolfedCore._prepare_polfed_initial_state(x)

    @test x == x0
    @test x_prepared' * x_prepared ≈ Matrix{Float64}(I, size(x0, 2), size(x0, 2))

    @test_throws ArgumentError Polfed.PolfedCore._prepare_polfed_initial_state(zeros(3))
    @test_throws DimensionMismatch Polfed.PolfedCore._prepare_polfed_initial_state(ones(2, 3))
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
