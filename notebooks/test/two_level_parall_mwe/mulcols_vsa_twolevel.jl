#!/usr/bin/env julia

using LinearAlgebra
using Pkg
using Printf
using Random
using SparseArrays
using Statistics
using Distributed

const ROOT_DIR = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(ROOT_DIR)

include(joinpath(ROOT_DIR, "src", "Polfed.jl"))
using .Polfed

# ------------------------------ fixed benchmark setup ------------------------------
const ORDER = 10000
const REPEATS = 5
const L = 20
const HX = 1.0
const SPIN = 0.5
const SEED = 1234

# Keep a local QREM type compatible with notebooks/test/QREM/cpu/construct_matrix.jl.
struct QREM
    name::String
    L::Int
    hx::Float64
    spin::Float64
    local_dim::Int
    diags::Vector{Float64}
    hilbertspacedim::Int

    function QREM(name::String, L::Int, hx::Float64, spin::Float64, diags::Vector{Float64})
        local_dim = Int(2spin + 1)
        hilbertspacedim = local_dim^L
        new(name, L, hx, spin, local_dim, diags, hilbertspacedim)
    end
end

include(joinpath(ROOT_DIR, "notebooks", "test", "QREM", "cpu", "construct_matrix.jl"))

# ------------------------------ custom optimized mapping ------------------------------
const Offdiagonal = Tuple{<:Number, Vector{Int}, Vector{Int}}
const Offdiagonals = Union{Offdiagonal, Vector{<:Offdiagonal}}

const UseThreadsInLoop = Dict(
    NoParallel => false,
    MulColsParallel => false,
    TwoLevelParallel => true,
)

function make_loop(use_threads::Bool)
    if use_threads
        return (range, f) -> (Threads.@threads for i in range; f(i); end)
    else
        return (range, f) -> (for i in range; f(i); end)
    end
end

@inline function mapping_state_i(
    X::AbstractVector{T},
    i::Int,
    diagonals::AbstractVector,
    val::Number,
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
) where {T<:Number}
    start = start_indices[i]
    @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i + 1] - 1
    sum_val = zero(T)
    for j in start:stop
        @inbounds sum_val += X[offdiags_flatten[j]]
    end
    @inbounds Y_i = diagonals[i] * X[i] + val * sum_val
    return Y_i
end

@inline function mapping_offdiagonals_state_i(
    X::AbstractVector{T},
    i::Int,
    val::Number,
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
) where {T<:Number}
    start = start_indices[i]
    @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i + 1] - 1
    sum_val = zero(T)
    for j in start:stop
        @inbounds sum_val += X[offdiags_flatten[j]]
    end
    return val * sum_val
end

function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
    loop::Function,
)
    (val, offdiags_flatten, start_indices) = offdiagonals
    loop(eachindex(X), @inline i -> begin
        Y_i = mapping_state_i(X, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds Y[i] = Y_i
    end)
end

function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Number, Vector{Int}, Vector{Int}}},
    loop::Function,
)
    @. Y = diagonals * X
    for (val, offdiags_flatten, start_indices) in offdiagonals
        loop(eachindex(X), @inline i -> begin
            Y_off_i = mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)
            @inbounds Y[i] += Y_off_i
        end)
    end
end

function mapping!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diagonals::AbstractVector,
    offdiagonals,
    loop::Function,
)
    @assert size(Y) == size(X)
    @assert length(diagonals) == size(X, 1)
    for col in axes(X, 2)
        mapping!(view(Y, :, col), view(X, :, col), diagonals, offdiagonals, loop)
    end
end

function optimized_mapping!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy)
    use_threads_in_loop = UseThreadsInLoop[typeof(parallel_strategy)]
    loop = make_loop(use_threads_in_loop)
    opt_map! = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> mapping!(Y, X, diagonals, offdiagonals, loop)
    return opt_map!
end

function install_custom_mapping_on_workers!(worker_ids::Vector{Int})
    isempty(worker_ids) && return

    @everywhere worker_ids begin
        if !isdefined(Main, :UseThreadsInLoop)
            const UseThreadsInLoop = Dict(
                Main.Polfed.NoParallel => false,
                Main.Polfed.MulColsParallel => false,
                Main.Polfed.TwoLevelParallel => true,
            )
        end

        if !isdefined(Main, :make_loop)
            function make_loop(use_threads::Bool)
                if use_threads
                    return (range, f) -> (Threads.@threads for i in range; f(i); end)
                else
                    return (range, f) -> (for i in range; f(i); end)
                end
            end
        end

        if !isdefined(Main, :mapping_state_i)
            @inline function mapping_state_i(
                X::AbstractVector{T},
                i::Int,
                diagonals::AbstractVector,
                val::Number,
                offdiags_flatten::Vector{Int},
                start_indices::Vector{Int},
            ) where {T<:Number}
                start = start_indices[i]
                @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i + 1] - 1
                sum_val = zero(T)
                for j in start:stop
                    @inbounds sum_val += X[offdiags_flatten[j]]
                end
                @inbounds Y_i = diagonals[i] * X[i] + val * sum_val
                return Y_i
            end
        end

        if !isdefined(Main, :mapping_offdiagonals_state_i)
            @inline function mapping_offdiagonals_state_i(
                X::AbstractVector{T},
                i::Int,
                val::Number,
                offdiags_flatten::Vector{Int},
                start_indices::Vector{Int},
            ) where {T<:Number}
                start = start_indices[i]
                @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i + 1] - 1
                sum_val = zero(T)
                for j in start:stop
                    @inbounds sum_val += X[offdiags_flatten[j]]
                end
                return val * sum_val
            end
        end

        if !isdefined(Main, :mapping!)
            function mapping!(
                Y::AbstractVector,
                X::AbstractVector,
                diagonals::AbstractVector,
                offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
                loop::Function,
            )
                (val, offdiags_flatten, start_indices) = offdiagonals
                loop(eachindex(X), @inline i -> begin
                    Y_i = mapping_state_i(X, i, diagonals, val, offdiags_flatten, start_indices)
                    @inbounds Y[i] = Y_i
                end)
            end

            function mapping!(
                Y::AbstractVector,
                X::AbstractVector,
                diagonals::AbstractVector,
                offdiagonals::Vector{<:Tuple{<:Number, Vector{Int}, Vector{Int}}},
                loop::Function,
            )
                @. Y = diagonals * X
                for (val, offdiags_flatten, start_indices) in offdiagonals
                    loop(eachindex(X), @inline i -> begin
                        Y_off_i = mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)
                        @inbounds Y[i] += Y_off_i
                    end)
                end
            end

            function mapping!(
                Y::AbstractMatrix,
                X::AbstractMatrix,
                diagonals::AbstractVector,
                offdiagonals,
                loop::Function,
            )
                @assert size(Y) == size(X)
                @assert length(diagonals) == size(X, 1)
                for col in axes(X, 2)
                    mapping!(view(Y, :, col), view(X, :, col), diagonals, offdiagonals, loop)
                end
            end
        end

        if !isdefined(Main, :optimized_mapping!)
            function optimized_mapping!(diagonals::AbstractVector, offdiagonals, parallel_strategy)
                use_threads_in_loop = UseThreadsInLoop[typeof(parallel_strategy)]
                loop = make_loop(use_threads_in_loop)
                return (Y, X) -> mapping!(Y, X, diagonals, offdiagonals, loop)
            end
        end
    end
end

# ------------------------------ benchmark logic ------------------------------
function usage()
    println("Usage: julia --project mulcols_vsa_twolevel.jl <nvec>")
    println("  nvec = number of mapped vectors/columns.")
    println("  MulCols mode: one thread per column (best when Julia -t equals nvec).")
    println("  TwoLevel mode: nworkers=nvec, threads_per_col=floor(Threads.nthreads()/nvec).")
    println("  Fixed setup: QREM(L=$L, hx=$HX, spin=$SPIN), Clenshaw order=$ORDER, repeats=$REPEATS")
end

function parse_nvec(args::Vector{String})
    any(==("--help"), args) && (usage(); exit(0))
    length(args) == 1 || error("Expected exactly one argument: <nvec>. Use --help.")
    nvec = parse(Int, args[1])
    nvec > 0 || error("nvec must be > 0.")
    return nvec
end

function build_qrem_problem(nvec::Int)
    D = Int(2SPIN + 1)^L
    rng_diag = MersenneTwister(SEED)
    diags = randn(rng_diag, D) .* sqrt(L / 2)
    qrem = QREM("qrem", L, HX, SPIN, diags)

    H = construct_matrix(qrem; pu="cpu")
    diagonals, offdiagonals = Polfed.PolfedCore.get_diags_and_offdiagonals_by_value(H)

    rng_vec = MersenneTwister(SEED + 1)
    X = randn(rng_vec, D, nvec)
    Y = zeros(Float64, D, nvec)

    coeffs = [1.0 / (k + 1) for k in 0:ORDER]

    return qrem, H, diagonals, offdiagonals, X, Y, coeffs
end

function make_b_storage(D::Int, nvec::Int)
    return [[zeros(Float64, D) for _ in 1:3] for _ in 1:nvec]
end

function run_mulcols(diagonals, offdiagonals, X, Y, coeffs)
    nvec = size(X, 2)
    D = size(X, 1)

    strategy = MulColsParallel()
    mapping! = optimized_mapping!(diagonals, offdiagonals, strategy)
    transform = Polfed.ClenshawMapping.Clenshaw(:Chebyshev, coeffs, ORDER, mapping!, D, Float64)
    b_storage = make_b_storage(D, nvec)

    Polfed.PolfedCore.clenshaw(transform, Y, X, b_storage, CPU(), strategy) # warm-up

    times = Float64[]
    for _ in 1:REPEATS
        t = @elapsed Polfed.PolfedCore.clenshaw(transform, Y, X, b_storage, CPU(), strategy)
        push!(times, t)
    end
    return times
end

function run_twolevel(diagonals, offdiagonals, X, Y, coeffs)
    nvec = size(X, 2)
    D = size(X, 1)
    nt_per_col = max(1, fld(Threads.nthreads(), nvec))

    strategy = TwoLevelParallel(nt_per_col)
    mapping! = optimized_mapping!(diagonals, offdiagonals, strategy)
    transform = Polfed.ClenshawMapping.Clenshaw(:Chebyshev, coeffs, ORDER, mapping!, D, Float64)
    b_storage = make_b_storage(D, nvec)

    setup_time = 0.0
    times = Float64[]

    try
        setup_time = @elapsed Polfed.PolfedCore.set_workers(X, strategy)
        install_custom_mapping_on_workers!(collect(strategy.worker_pool.workers))
        Polfed.PolfedCore.clenshaw(transform, Y, X, b_storage, CPU(), strategy) # warm-up

        for _ in 1:REPEATS
            t = @elapsed Polfed.PolfedCore.clenshaw(transform, Y, X, b_storage, CPU(), strategy)
            push!(times, t)
        end
    finally
        Polfed.PolfedCore.remove_workers(strategy)
    end

    return nt_per_col, setup_time, times
end

function summarize_times(label::String, times::Vector{Float64})
    @printf("  %s min  = %.6f s\n", label, minimum(times))
    @printf("  %s mean = %.6f s\n", label, mean(times))
    @printf("  %s std  = %.6f s\n", label, std(times))
end

function main(args::Vector{String})
    nvec = parse_nvec(args)
    qrem, H, diagonals, offdiagonals, X, Y, coeffs = build_qrem_problem(nvec)

    println("Benchmark: MulCols vs TwoLevel for QREM custom mapping")
    println("  L = $(qrem.L)")
    println("  D = $(qrem.hilbertspacedim)")
    println("  nnz(H) = $(nnz(H))")
    println("  nvec = $nvec")
    println("  Julia threads = $(Threads.nthreads())")
    println("  Clenshaw order = $ORDER")
    println("  repeats = $REPEATS")

    if nvec != Threads.nthreads()
        @warn "For strict one-thread-per-column in MulCols, launch Julia with -t nvec." nvec=nvec julia_threads=Threads.nthreads()
    end

    mulcols_times = run_mulcols(diagonals, offdiagonals, X, copy(Y), coeffs)
    nt_per_col, setup_time, twolevel_times = run_twolevel(diagonals, offdiagonals, X, copy(Y), coeffs)

    println()
    println("MulColsParallel (1 thread per column mapping):")
    summarize_times("MulCols", mulcols_times)

    println()
    println("TwoLevelParallel (workers + pmap):")
    println("  threads per column (auto) = $nt_per_col")
    @printf("  worker setup time = %.6f s\n", setup_time)
    summarize_times("TwoLevel", twolevel_times)

    speedup = mean(mulcols_times) / mean(twolevel_times)
    println()
    @printf("Speedup (MulCols_mean / TwoLevel_mean) = %.4f\n", speedup)
end

main(ARGS)
