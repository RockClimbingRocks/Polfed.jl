using SparseArrays
using LinearAlgebra
using Random
using Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed

function build_complex_hermitian(n::Int, density::Float64; seed::Int = 1234)
    rng = MersenneTwister(seed)

    re_part = sprand(rng, n, n, density)
    im_part = sprand(rng, n, n, density)
    A = SparseMatrixCSC{ComplexF64, Int}(re_part .+ (1im) .* im_part)

    # Hermitian by construction: H = (A + A') / 2
    H = (A + A') / 2
    return H
end

function run_case(label::AbstractString, f::Function)
    println("\n== ", label, " ==")
    GC.gc()
    f() # warm-up
    GC.gc()

    vals = nothing
    vecs = nothing
    t = @elapsed begin
        vals, vecs = f()
    end

    println("vals eltype: ", eltype(vals), ", vecs eltype: ", eltype(vecs))
    println("returned eigenpairs: ", length(vals))
    println("elapsed: ", round(t; digits=4), " s")
    return vals, vecs, t
end

run_case(f::Function, label::AbstractString) = run_case(label, f)

function residual_stats(H::SparseMatrixCSC{ComplexF64, Int}, vals, vecs)
    k = min(length(vals), size(vecs, 2))
    max_residual = 0.0
    max_eig_imag = 0.0

    for i in 1:k
        lambda = vals[i]
        max_eig_imag = max(max_eig_imag, abs(imag(lambda)))

        v = vecs[:, i]
        r = H * v - lambda * v
        max_residual = max(max_residual, norm(r))
    end

    return max_residual, max_eig_imag
end

function compare_eigs(vals_a, vals_b)
    a = sort(real.(vals_a))
    b = sort(real.(vals_b))
    n = min(length(a), length(b))
    n == 0 && return NaN
    return maximum(abs.(a[1:n] .- b[1:n]))
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
    howmany = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 6
    target = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
    density = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.03
    run_optimized = length(ARGS) >= 5 ? parse(Int, ARGS[5]) != 0 : false

    println("Building complex Hermitian sparse matrix...")
    H = build_complex_hermitian(n, density)

    herm_defect = norm(Matrix(H - H'))
    println("size: ", size(H), ", density: ", density)
    @printf("Hermitian defect ||H-H'|| = %.3e\n", herm_defect)

    Random.seed!(1)
    x0 = randn(ComplexF64, n)
    x0 ./= norm(x0)

    fact = FactorizationConfig(
        tol = 1e-8,
        eigentol = 1e-7,
        overestimate_iters = 1.0,
    )

    dos = DoSConfig(
        N = 64,
        R = 20,
        kernel = :Jackson,
    )

    mapping_base = MappingConfig(
        parallel_strategy = NoParallel(),
        optimize_mapping = false,
    )

    vals_base, vecs_base, _ = run_case("Complex Hermitian (optimize_mapping=false)") do
        polfed(
            H,
            x0,
            howmany,
            target;
            produce_report = false,
            mapping = mapping_base,
            fact = fact,
            dos = dos,
        )
    end

    res_base, imag_base = residual_stats(H, vals_base, vecs_base)

    println("\n== Diagnostics ==")
    @printf("baseline max residual norm: %.3e\n", res_base)
    @printf("baseline max |Im(lambda)|: %.3e\n", imag_base)

    if run_optimized
        mapping_opt = MappingConfig(
            parallel_strategy = NoParallel(),
            optimize_mapping = true,
        )

        vals_opt, vecs_opt, _ = run_case("Complex Hermitian (optimize_mapping=true)") do
            polfed(
                H,
                x0,
                howmany,
                target;
                produce_report = false,
                mapping = mapping_opt,
                fact = fact,
                dos = dos,
            )
        end

        res_opt, imag_opt = residual_stats(H, vals_opt, vecs_opt)
        eig_diff = compare_eigs(vals_base, vals_opt)

        @printf("optimized max residual norm: %.3e\n", res_opt)
        @printf("optimized max |Im(lambda)|: %.3e\n", imag_opt)
        @printf("max |lambda_baseline - lambda_optimized|: %.3e\n", eig_diff)
    else
        println("optimized mapping check: skipped (pass arg5=1 to enable)")
    end
end

main()
