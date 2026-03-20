using SparseArrays, LinearAlgebra, Random, Printf, UnPack

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed

const QREM_ROOT = joinpath(@__DIR__, "QREM")
include(joinpath(QREM_ROOT, "QREM.jl"))

const PD = Polfed.PolfedDefaults

function parse_int_list(s::AbstractString)
    isempty(strip(s)) && return Int[]
    return [parse(Int, strip(x)) for x in split(s, ",")]
end

function make_qrem(L::Int, hx::Float64, spin::Float64, avgs::Int)
    params = Dict{Symbol, Any}(
        :model_name => "qrem",
        :L => L,
        :hx => hx,
        :spin => spin,
        :avgs => avgs,
        :runname => "verbosity_demo",
    )
    return construct_model(params)
end

function run_case(L::Int, level::Int; howmany::Int = 4, target::Real = 0.0, hx::Float64 = 1.0, spin::Float64 = 0.5, avgs::Int = 0)
    PD.verbosity[] = level

    qrem = make_qrem(L, hx, spin, avgs)
    mat = construct_matrix(qrem; pu = "cpu")

    dim = size(mat, 1)
    x0 = randn(dim)
    x0 ./= norm(x0)

    mapping = MappingConfig(
        parallel_strategy = NoParallel(),
        optimize_mapping = false,
    )

    transform = TransformConfig(
        order = nothing, # auto-select order
    )

    fact = FactorizationConfig(
        overestimate_iters = 1.0,
        tol = 1e-12,
        eigentol = 1e-9,
    )

    # Keep DoS light so the demo runs quickly.
    dos = DoSConfig(
        N = 36,
        R = 8,
        kernel = :Jackson,
    )

    k = min(howmany, max(1, dim ÷ 32))
    vals = nothing
    elapsed = @elapsed begin
        vals, _ = polfed(
            mat,
            x0,
            k,
            target;
            produce_report = false,
            mapping = mapping,
            transform = transform,
            fact = fact,
            dos = dos,
        )
    end

    # println(@sprintf("L=%2d  dim=%6d  verbosity=%d  howmany=%d  time=%.3fs  first_eval=%.6f", L, dim, level, k, elapsed, vals[1]))
    return nothing
end

function main()
    sizes = length(ARGS) >= 1 ? parse_int_list(ARGS[1]) : 10
    levels = length(ARGS) >= 2 ? parse_int_list(ARGS[2]) : [0, 1, 2, 3]
    howmany = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
    target = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.0
    hx = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.0
    spin = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.5

    previous = PD.verbosity[]

    try
        println("Running POLFED verbosity demo")
        println("sizes     = ", sizes)
        println("levels    = ", levels)
        println("howmany   = ", howmany)
        println("target    = ", target)
        println("hx        = ", hx)
        println("spin      = ", spin)
        println()

        for level in levels
            println("=== Verbosity Level ", level, " ===")
            for L in sizes
                run_case(L, level; howmany = howmany, target = target, hx = hx, spin = spin)
            end
            println()
        end
    finally
        PD.verbosity[] = previous
    end
end

main()
