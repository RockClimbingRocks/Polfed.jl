using SparseArrays, LinearAlgebra, BenchmarkTools, Random
using UnPack

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed

const QREM_ROOT = joinpath(@__DIR__, "QREM")
include(joinpath(QREM_ROOT, "QREM.jl"))


function run_case(label::AbstractString, f::Function)
    println("\n== ", label, " ==")
    GC.gc()
    f() # warm-up to compile
    GC.gc()
    @time f()
    # show(stdout, MIME("text/plain"), trial)
    println()
    return nothing
end

run_case(f::Function, label::AbstractString) = run_case(label, f)


function make_qrem(L::Int, hx::Float64, spin::Float64, avgs::Int)
    params = Dict{Symbol, Any}(
        :model_name => "qrem",
        :L => L,
        :hx => hx,
        :spin => spin,
        :avgs => avgs,
        :runname => "bench",
    )
    return construct_model(params)
end


function lanczos_extrema(f!, x0)
    Emin = first(collect(Polfed.Lanczos.lanczos(f!, x0, 1; which=:SR, maxdim=1000)[1]))
    Emax = last(collect(Polfed.Lanczos.lanczos(f!, x0, 1; which=:LR,  maxdim=1000)[1]))
    return Emin, Emax
end


function main()
    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    howmany = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    target = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
    hx = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0

    spin = 0.5
    avgs = 0

    println("Building QREM model...")
    qrem = make_qrem(L, hx, spin, avgs)

    mat = construct_matrix(qrem; pu="cpu")
    dim = size(mat, 1)
    println("Hilbert space dim: ", dim)

    Random.seed!(1)
    x0 = rand(dim)
    x0 ./= norm(x0)

    # Case 1: plain sparse matrix
    run_case("Sparse matrix (baseline)") do
        polfed(mat, x0, howmany, target;
            mapping=MappingConfig(optimize_mapping=false),
            produce_report=false
        )
    end

    # Case 2: sparse matrix + optimize_mapping
    run_case("Sparse matrix + optimize_mapping=true") do
        polfed(mat, x0, howmany, target;
            mapping=MappingConfig(optimize_mapping=true),
            produce_report=false
        )
    end

    # Case 3: optimized mapping from QREM (rescaled)
    map_cpu = map_vec(qrem; a=1.0, b=0.0, pu="cpu")
    Emin, Emax = lanczos_extrema(map_cpu, x0)
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2
    f!_rescaled_cpu = map_vec(qrem; a=a, b=b, pu="cpu")

    # Case 3: optimized mapping from QREM (rescaled)
    mapping_rescaled = MappingConfig(
        parallel_strategy=MulColsParallel(),
        f!_rescaled=f!_rescaled_cpu,
        Emin=Emin,
        Emax=Emax,
    )

    run_case("Optimized mapping from QREM (rescaled)") do
        polfed(map_cpu, x0, howmany, target;
            mapping=mapping_rescaled,
            produce_report=false
        )
    end

    # Case 5: optimized mapping + Clenshaw kernels from QREM (CPU)
    crr, cfs = clenshaw(qrem; a=a, b=b, pu="cpu")

    mapping_clenshaw = MappingConfig(
        parallel_strategy=MulColsParallel(),
        f!_rescaled=f!_rescaled_cpu,
        clenshaw_recurrence=crr,
        clenshaw_finalsum=cfs,
        Emin=Emin,
        Emax=Emax,
    )

    run_case("Optimized mapping + Clenshaw kernels (QREM CPU)") do
        polfed(map_cpu, x0, howmany, target;
            mapping=mapping_clenshaw,
            produce_report=false
        )
    end
end

main()
