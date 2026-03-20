#!/usr/bin/env julia

using Dates
using HDF5
using LinearAlgebra
using Printf
using Random
using SparseArrays

include(joinpath(@__DIR__, "..", "..", "..", "src", "Polfed.jl"))
using .Polfed

const DEFAULT_MODE = "both"
const DEFAULT_L = 14
const DEFAULT_DELTA = 0.123
const DEFAULT_HOWMANY = 20
const DEFAULT_TARGET = 0.0
const DEFAULT_INITIAL_VECTORS = 4
const DEFAULT_MULCOLS_THREADS_PER_COL = 1
const DEFAULT_TWOLEVEL_THREADS_PER_COL = 2
const DEFAULT_SEED = 1234
const DEFAULT_OUTPUT_H5 = joinpath(@__DIR__, "twolevelparallelization_results.h5")

function print_usage()
    println(
        """
        Usage:
          julia --project=. notebooks/test/two_level_parall/twolevelparallelization.jl [options]

        Options:
          --mode=both|mulcols|twolevel
          --L=<Int>
          --delta=<Float64>
          --howmany=<Int>
          --target=<Float64>
          --initial-vectors=<Int>
          --mulcols-threads-per-col=<Int>
          --twolevel-threads-per-col=<Int>
          --seed=<Int>
          --output-h5=<path>
          --help

        Example:
          julia --project=. notebooks/test/two_level_parall/twolevelparallelization.jl \\
            --mode=both \\
            --initial-vectors=6 \\
            --twolevel-threads-per-col=3 \\
            --output-h5=notebooks/test/two_level_parall/run.h5
        """
    )
end

parse_int_positive(name::String, x::AbstractString) = begin
    value = try
        parse(Int, x)
    catch
        error("Could not parse `$name` from `$x`.")
    end
    value > 0 || error("`$name` must be > 0, got $value.")
    value
end

parse_float(name::String, x::AbstractString) = try
    parse(Float64, x)
catch
    error("Could not parse `$name` from `$x`.")
end

function parse_cli(args::Vector{String})
    mode = DEFAULT_MODE
    L = DEFAULT_L
    delta = DEFAULT_DELTA
    howmany = DEFAULT_HOWMANY
    target = DEFAULT_TARGET
    initial_vectors = DEFAULT_INITIAL_VECTORS
    mulcols_threads_per_col = DEFAULT_MULCOLS_THREADS_PER_COL
    twolevel_threads_per_col = DEFAULT_TWOLEVEL_THREADS_PER_COL
    seed = DEFAULT_SEED
    output_h5 = DEFAULT_OUTPUT_H5

    for arg in args
        arg == "--help" && (print_usage(); exit(0))

        if startswith(arg, "--mode=")
            mode = split(arg, "=", limit=2)[2]
            mode in ("both", "mulcols", "twolevel") || error("Unsupported mode `$mode`.")
        elseif startswith(arg, "--L=")
            L = parse_int_positive("L", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--delta=")
            delta = parse_float("delta", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--howmany=")
            howmany = parse_int_positive("howmany", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--target=")
            target = parse_float("target", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--initial-vectors=")
            initial_vectors = parse_int_positive("initial-vectors", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--mulcols-threads-per-col=")
            mulcols_threads_per_col = parse_int_positive("mulcols-threads-per-col", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--twolevel-threads-per-col=")
            twolevel_threads_per_col = parse_int_positive("twolevel-threads-per-col", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--seed=")
            seed = parse_int_positive("seed", split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--output-h5=")
            output_h5 = String(split(arg, "=", limit=2)[2])
        else
            error("Unknown argument `$arg`. Use --help for usage.")
        end
    end

    return (
        mode = mode,
        L = L,
        delta = delta,
        howmany = howmany,
        target = target,
        initial_vectors = initial_vectors,
        mulcols_threads_per_col = mulcols_threads_per_col,
        twolevel_threads_per_col = twolevel_threads_per_col,
        seed = seed,
        output_h5 = output_h5,
    )
end

function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int=L ÷ 2)
    basis = [b for b in 0:2^L-1 if count_ones(b) == Nup]
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (col, state) in enumerate(basis)
        for i in 1:L
            j = i % L + 1
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1

            szsz = (0.5 - si) * (0.5 - sj)
            push!(rows, col); push!(cols, col); push!(vals, delta * szsz)

            if si != sj
                flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                if haskey(bmap, flipped)
                    push!(rows, bmap[flipped]); push!(cols, col); push!(vals, 0.5)
                end
            end
        end
    end

    return sparse(rows, cols, vals, dim, dim)
end

function make_initial_block(dim::Int, nvec::Int; seed::Int)
    nvec <= dim || error("initial-vectors must be <= Hilbert-space dimension ($dim), got $nvec.")
    rng = MersenneTwister(seed)
    x0_raw = randn(rng, dim, nvec)
    q = qr(x0_raw).Q
    return Matrix(q[:, 1:nvec])
end

function run_case(
    label::String,
    H::SparseMatrixCSC{Float64,Int},
    x0::Matrix{Float64},
    howmany::Int,
    target::Float64,
    parallel_strategy,
)
    mapping = Polfed.MappingConfig(parallel_strategy=parallel_strategy)
    elapsed = @elapsed vals, vecs, report = Polfed.polfed(
        H, x0, howmany, target;
        produce_report=true,
        mapping=mapping,
    )

    timing = report.benchmark.timings
    show_n = min(length(vals), 5)
    vals_show = collect(vals[1:show_n])

    println()
    println("[$label]")
    println("  strategy: $(typeof(parallel_strategy))")
    println("  total walltime (report): $(round(timing.total_wt; digits=4)) s")
    println("  elapsed walltime (script): $(round(elapsed; digits=4)) s")
    println("  eigenvalues returned: $(length(vals))")
    println("  first $(show_n) eigenvalues: $(vals_show)")

    return (
        label = label,
        strategy_type = string(typeof(parallel_strategy)),
        elapsed = elapsed,
        vals = vals,
        report = report,
    )
end

function save_results_h5(output_h5::AbstractString, cfg, dim::Int, results::Vector{NamedTuple})
    mkpath(dirname(output_h5))

    h5open(output_h5, "w") do h5
        g_cfg = create_group(h5, "config")
        g_cfg["created_at"] = string(now())
        g_cfg["mode"] = cfg.mode
        g_cfg["L"] = cfg.L
        g_cfg["delta"] = cfg.delta
        g_cfg["howmany"] = cfg.howmany
        g_cfg["target"] = cfg.target
        g_cfg["initial_vectors"] = cfg.initial_vectors
        g_cfg["mulcols_threads_per_col"] = cfg.mulcols_threads_per_col
        g_cfg["twolevel_threads_per_col"] = cfg.twolevel_threads_per_col
        g_cfg["seed"] = cfg.seed
        g_cfg["julia_threads_main"] = Threads.nthreads()
        g_cfg["hilbert_dimension"] = dim

        g_runs = create_group(h5, "runs")
        for r in results
            key = lowercase(replace(r.label, " " => "_"))
            g_run = create_group(g_runs, key)
            timing = r.report.benchmark.timings

            g_run["label"] = r.label
            g_run["strategy_type"] = r.strategy_type
            g_run["elapsed_walltime_script_s"] = r.elapsed
            g_run["total_walltime_report_s"] = timing.total_wt
            g_run["total_cputime_report_s"] = timing.total_ct
            g_run["factorization_walltimes_s"] = collect(timing.fact_wt)
            g_run["factorization_cputimes_s"] = collect(timing.fact_ct)
            g_run["num_eigenvalues"] = length(r.vals)
            g_run["eigenvalues"] = collect(r.vals)
        end
    end
end

function main(args::Vector{String})
    cfg = parse_cli(args)

    println("Julia threads on main process: $(Threads.nthreads())")
    println("Configuration:")
    println("  mode = $(cfg.mode)")
    println("  L = $(cfg.L)")
    println("  delta = $(cfg.delta)")
    println("  howmany = $(cfg.howmany)")
    println("  target = $(cfg.target)")
    println("  initial_vectors = $(cfg.initial_vectors)")
    println("  mulcols_threads_per_col = $(cfg.mulcols_threads_per_col)")
    println("  twolevel_threads_per_col = $(cfg.twolevel_threads_per_col)")
    println("  seed = $(cfg.seed)")
    println("  output_h5 = $(cfg.output_h5)")

    H = construct_xxz_spin_sector(cfg.L, cfg.delta)
    dim = size(H, 1)
    println("Hilbert-space dimension: $dim")

    x0 = make_initial_block(dim, cfg.initial_vectors; seed=cfg.seed)

    vals_mul = nothing
    vals_two = nothing
    results = NamedTuple[]

    if cfg.mode in ("both", "mulcols")
        strategy = Polfed.MulColsParallel(cfg.mulcols_threads_per_col)
        run_mul = run_case("MulColsParallel", H, x0, cfg.howmany, cfg.target, strategy)
        vals_mul = run_mul.vals
        push!(results, run_mul)
    end

    if cfg.mode in ("both", "twolevel")
        strategy = Polfed.TwoLevelParallel(cfg.twolevel_threads_per_col)
        run_two = run_case("TwoLevelParallel", H, x0, cfg.howmany, cfg.target, strategy)
        vals_two = run_two.vals
        push!(results, run_two)
    end

    if cfg.mode == "both" && vals_mul !== nothing && vals_two !== nothing
        n = min(length(vals_mul), length(vals_two))
        λ_mul = sort(collect(vals_mul))[1:n]
        λ_two = sort(collect(vals_two))[1:n]
        max_abs_diff = maximum(abs.(λ_mul .- λ_two))
        println()
        @printf("Max |λ_mulcols - λ_twolevel| over first %d sorted values: %.6e\n", n, max_abs_diff)
    end

    save_results_h5(cfg.output_h5, cfg, dim, results)
    println("Saved timings + configuration to: $(abspath(cfg.output_h5))")
end

main(ARGS)
