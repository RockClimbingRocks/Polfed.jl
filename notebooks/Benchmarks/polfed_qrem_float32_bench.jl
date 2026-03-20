using SparseArrays
using LinearAlgebra
using Random, UnPack
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed

const QREM_ROOT = joinpath(@__DIR__, "QREM")
include(joinpath(QREM_ROOT, "QREM.jl"))

function make_qrem(L::Int, hx::Float64, spin::Float64, avgs::Int)
    params = Dict{Symbol, Any}(
        :model_name => "qrem",
        :L => L,
        :hx => hx,
        :spin => spin,
        :avgs => avgs,
        :runname => "float32-vs-float64-bench",
    )
    return construct_model(params)
end

function run_case(label::AbstractString, f::Function)
    println("\n== ", label, " ==")
    GC.gc()
    f() # warm-up / compilation
    GC.gc()

    vals = nothing
    vecs = nothing
    t = @elapsed begin
        vals, vecs = f()
    end

    println("vals eltype: ", eltype(vals), ", vecs eltype: ", eltype(vecs))
    println("all finite vals: ", all(isfinite, vals), ", all finite vecs: ", all(isfinite, vecs))
    println("returned eigenpairs: ", length(vals))
    println("elapsed: ", round(t; digits=4), " s")

    return vals, vecs, t
end

run_case(f::Function, label::AbstractString) = run_case(label, f)

function compare_vals(vals64, vals32)
    s64 = sort(Float64.(vals64))
    s32 = sort(Float64.(vals32))

    n = min(length(s64), length(s32))
    if n == 0
        println("\nNo eigenvalues to compare.")
        return
    end

    if length(s64) != length(s32)
        println("\nWarning: different number of eigenvalues returned (Float64=$(length(s64)), Float32=$(length(s32))).")
    end

    d = abs.(s64[1:n] .- s32[1:n])
    max_abs = maximum(d)
    denom = max(maximum(abs.(s64[1:n])), eps(Float64))
    max_rel = max_abs / denom

    println("\n== Eigenvalue Comparison (Float64 vs Float32) ==")
    @printf("max |Δλ| = %.6e\n", max_abs)
    @printf("max relative |Δλ| = %.6e\n", max_rel)

    println("\nIndex  Float64              Float32              |Δ|")
    for i in 1:n
        @printf("%5d  % .12e  % .12e  %.6e\n", i, s64[i], s32[i], d[i])
    end
end

function main()
    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    howmany = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    target64 = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
    hx = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0

    target32 = Float32(target64)
    spin = 0.5
    avgs = 0

    println("Building QREM model...")
    qrem = make_qrem(L, hx, spin, avgs)

    mat64 = construct_matrix(qrem; pu="cpu")
    mat32 = SparseMatrixCSC{Float32, Int}(mat64)
    dim = size(mat64, 1)

    println("Hilbert space dim: ", dim)
    println("Matrix element types: Float64 -> ", eltype(mat64), ", Float32 -> ", eltype(mat32))

    Random.seed!(1)
    x0base = randn(Float64, dim)
    x0base ./= norm(x0base)

    x064 = copy(x0base)
    x032 = Float32.(x0base)
    x032 ./= norm(x032)

    mapping64 = MappingConfig(
        parallel_strategy = MulColsParallel(),
        optimize_mapping = true,
    )

    mapping32 = MappingConfig(
        parallel_strategy = MulColsParallel(),
        optimize_mapping = true,
    )

    fact64 = FactorizationConfig(
        tol = 1e-9,
        eigentol = 1e-8,
        overestimate_iters = 1.0,
    )

    # Relaxed tolerances for Float32.
    fact32 = FactorizationConfig(
        tol = 1f-5,
        eigentol = 1f-4,
        overestimate_iters = 1.0,
    )

    dos = DoSConfig(
        N = 64,
        R = 20,
        kernel = :Jackson,
    )

    vals64, _, _ = run_case("QREM Float64 matrix") do
        polfed(
            mat64,
            x064,
            howmany,
            target64;
            produce_report = false,
            mapping = mapping64,
            fact = fact64,
            dos = dos,
        )
    end

    vals32, _, _ = run_case("QREM Float32 matrix") do
        polfed(
            mat32,
            x032,
            howmany,
            target32;
            produce_report = false,
            mapping = mapping32,
            fact = fact32,
            dos = dos,
        )
    end

    compare_vals(vals64, vals32)
end

main()
