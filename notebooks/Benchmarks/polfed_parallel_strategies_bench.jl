using SparseArrays, LinearAlgebra, BenchmarkTools, Random, Printf
using Base.Threads: nthreads

# Example:
# julia --project -t 16 notebooks/Benchmarks/polfed_parallel_strategies_bench.jl 18 64 4 0.0 3 96 12

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed


function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
    basis = [b for b in 0:(2^L - 1) if count_ones(b) == Nup]
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))
    rows, cols, vals = Int[], Int[], Float64[]

    for (col, state) in enumerate(basis)
        for i in 1:L
            j = i % L + 1
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1
            szz = (0.5 - si) * (0.5 - sj)
            push!(rows, col); push!(cols, col); push!(vals, delta * szz)

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


make_block_seed(dim::Int, blocksize::Int; seed::Int = 1) = begin
    Random.seed!(seed)
    Matrix(qr(rand(dim, blocksize)).Q)
end


function trial_median_ns(trial::BenchmarkTools.Trial)
    ts = sort!(copy(trial.times))
    return ts[cld(length(ts), 2)]
end


function print_trial(label::AbstractString, trial::BenchmarkTools.Trial)
    ts = sort!(copy(trial.times))
    min_ns = ts[1]
    med_ns = ts[cld(length(ts), 2)]
    mean_ns = sum(ts) / length(ts)
    min_entry = minimum(trial)
    @printf(
        "%-28s min=%9.3f ms  med=%9.3f ms  mean=%9.3f ms  alloc=%8.2f KiB  allocs=%d\n",
        label,
        min_ns / 1e6,
        med_ns / 1e6,
        mean_ns / 1e6,
        min_entry.memory / 1024,
        min_entry.allocs,
    )
end


function strategy_label(strategy)
    strategy isa NoParallel && return "NoParallel()"
    strategy isa MulColsParallel && return "MulColsParallel($(strategy.nt_per_col))"
    strategy isa TwoLevelParallel && return "TwoLevelParallel($(strategy.nt_per_col))"
    return string(typeof(strategy))
end


function solve_case(
    mat,
    x0,
    howmany::Int,
    target,
    dos_cfg::DoSConfig,
    fact_cfg::FactorizationConfig,
    strategy_factory::Function;
    seed::Int,
)
    Random.seed!(seed)
    mapping = MappingConfig(
        optimize_mapping = true,
        parallel_strategy = strategy_factory(),
    )

    vals, vecs = polfed(
        mat,
        x0,
        howmany,
        target;
        mapping = mapping,
        dos = dos_cfg,
        fact = fact_cfg,
        produce_report = false,
    )

    return vals, vecs
end


function compare_vals(reference::AbstractVector, candidate::AbstractVector; atol=1e-8, rtol=1e-8)
    same_length = length(reference) == length(candidate)
    n = min(length(reference), length(candidate))
    n == 0 && return nothing, nothing

    ref_view = view(reference, 1:n)
    cand_view = view(candidate, 1:n)
    max_abs_err = maximum(abs.(ref_view .- cand_view))
    matches = same_length && isapprox(ref_view, cand_view; atol=atol, rtol=rtol)

    return matches, max_abs_err
end


function run_case(label::AbstractString, f::Function; samples::Int, evals::Int)
    GC.gc()
    f()
    GC.gc()
    trial = @benchmark $f() samples=samples evals=evals
    print_trial(label, trial)
    return trial
end


function main()
    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 16
    howmany = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32
    blocksize = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
    target = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.0
    samples = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
    dos_N = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 96
    dos_R = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 12

    available_threads = nthreads()
    available_threads < blocksize && error(
        "Block size $(blocksize) exceeds Julia thread count $(available_threads). " *
        "Run with at least blocksize Julia threads."
    )

    nt_per_col = max(fld(available_threads, blocksize), 1)
    effective_parallel_cpus = blocksize * nt_per_col
    delta = 0.123
    seed = 1

    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    Polfed.PolfedDefaults.verbosity[] = Polfed.PolfedDefaults.POLFED_SILENT_LEVEL

    try
        println("Building XXZ benchmark instance...")
        mat = construct_xxz_spin_sector(L, delta, L ÷ 2)
        dim = size(mat, 1)
        x0 = make_block_seed(dim, blocksize; seed=seed)
        dos_cfg = DoSConfig(N=dos_N, R=dos_R)
        fact_cfg = FactorizationConfig()

        println("Hilbert space dim           : ", dim)
        println("Julia threads available     : ", available_threads)
        println("Block size                  : ", blocksize)
        println("Threads per column          : ", nt_per_col)
        println("Parallel CPU budget         : ", effective_parallel_cpus)
        println("DoS moments / vectors       : ", dos_N, " / ", dos_R)
        println("Benchmark samples / evals   : ", samples, " / 1")

        if effective_parallel_cpus != available_threads
            println("Note: thread count is not divisible by block size; parallel cases use ", effective_parallel_cpus, " worker threads.")
        end
        if nt_per_col == 1
            println("Note: nt_per_col == 1, so MulColsParallel stays in the per-column mode.")
        else
            println("Note: MulColsParallel($(nt_per_col)) uses the new single-process row-chunk path.")
        end
        println("Note: TwoLevelParallel($(nt_per_col)) uses ", blocksize, " workers x ", nt_per_col, " threads, plus the main coordinator process.")

        cases = [
            ("NoParallel()", () -> NoParallel()),
            ("MulColsParallel($(nt_per_col))", () -> MulColsParallel(nt_per_col)),
            ("TwoLevelParallel($(nt_per_col))", () -> TwoLevelParallel(nt_per_col)),
        ]

        println("\nCorrectness check:")
        ref_vals, _ = solve_case(mat, x0, howmany, target, dos_cfg, fact_cfg, cases[1][2]; seed=seed)
        println("  ", rpad(cases[1][1], 26), " converged=", length(ref_vals))

        for (label, strategy_factory) in Iterators.drop(cases, 1)
            vals, _ = solve_case(mat, x0, howmany, target, dos_cfg, fact_cfg, strategy_factory; seed=seed)
            matches, max_abs_err = compare_vals(ref_vals, vals)
            if isnothing(matches)
                @printf("  %-26s converged=%d  match=n/a  max|dv|=n/a\n", label, length(vals))
            else
                @printf(
                    "  %-26s converged=%d  match=%s  max|dv|=%.3e\n",
                    label,
                    length(vals),
                    string(matches),
                    max_abs_err,
                )
            end
        end

        println("\nBenchmark results:")
        trials = Dict{String, BenchmarkTools.Trial}()
        for (label, strategy_factory) in cases
            bench = () -> begin
                solve_case(mat, x0, howmany, target, dos_cfg, fact_cfg, strategy_factory; seed=seed)
                return nothing
            end
            trials[label] = run_case(label, bench; samples=samples, evals=1)
        end

        ref_trial = trials[cases[1][1]]
        ref_med = trial_median_ns(ref_trial)
        println("\nRelative speedups vs NoParallel():")
        for (label, _) in Iterators.drop(cases, 1)
            speedup = ref_med / trial_median_ns(trials[label])
            @printf("  %-26s %.3fx\n", label, speedup)
        end
    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
end


main()
