#!/usr/bin/env julia

using LinearAlgebra
using Pkg
using Random
using SparseArrays

const ROOT_DIR = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(ROOT_DIR)

include(joinpath(@__DIR__, "..", "..", "..", "src", "Polfed.jl"))
using .Polfed

function usage()
    println("Usage: julia --project=. notebooks/test/two_level_parall/run_mulcols_parallel.jl <L> [howmany] [target] [delta] [seed]")
    println("Example: julia --project=. notebooks/test/two_level_parall/run_mulcols_parallel.jl 14 20 0.0 0.123 1234")
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

function make_x0(dim::Int, nvec::Int, seed::Int)
    nvec <= dim || error("Number of columns ($nvec) must be <= Hilbert-space dimension ($dim).")
    rng = MersenneTwister(seed)
    x0_raw = randn(rng, dim, nvec)
    q = qr(x0_raw).Q
    return Matrix(q[:, 1:nvec])
end

function main(args::Vector{String})
    length(args) >= 1 || (usage(); error("Need <L>."))

    L = parse(Int, args[1])
    howmany = length(args) >= 2 ? parse(Int, args[2]) : 1500
    target = length(args) >= 3 ? parse(Float64, args[3]) : 0.0
    delta = length(args) >= 4 ? parse(Float64, args[4]) : 0.123
    seed = length(args) >= 5 ? parse(Int, args[5]) : 1234

    L > 0 || error("L must be > 0.")
    howmany > 0 || error("howmany must be > 0.")

    H = construct_xxz_spin_sector(L, delta)
    dim = size(H, 1)
    ncols = max(1, Threads.nthreads())
    x0 = make_x0(dim, ncols, seed)

    mapping = Polfed.MappingConfig(optimize_mapping=true, parallel_strategy=Polfed.MulColsParallel(1))

    println("MulColsParallel run")
    println("  L = $L")
    println("  Hilbert dimension = $dim")
    println("  Julia threads = $(Threads.nthreads())")
    println("  initial vector columns = $ncols (same as Julia threads)")
    println("  howmany = $howmany, target = $target")

    elapsed = @elapsed vals, vecs, report = Polfed.polfed(
        H, x0, howmany, target;
        produce_report=true,
        mapping=mapping,
    )

    println("  report total walltime = $(round(report.benchmark.timings.total_wt; digits=4)) s")
    println("  script elapsed walltime = $(round(elapsed; digits=4)) s")
    show_n = min(length(vals), 5)
    println("  first $(show_n) eigenvalues = $(collect(vals[1:show_n]))")
end

main(ARGS)
