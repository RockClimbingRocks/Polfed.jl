using Polfed
using BenchmarkTools
using LinearAlgebra
using Polyester: @batch
using Printf
using Random
using StaticArrays: SVector
using Statistics: median

include("../qrem.jl")

# Compare three ways of rotating the three Clenshaw work buffers:
#
#   1. permute!  — the implementation currently used by Clenshaw;
#   2. local refs — rotate three local references, without mutating the workspace vector;
#   3. unrolled3 — perform three recurrence steps in the physical buffers, so their
#                  orientation returns to the starting orientation after every triple.
#
# This benchmark uses the real QREM spin_flip/packed mapping and the Chebyshev recurrence
#
#   b_k = c_k X + 2 A b_{k+1} - b_{k+2}.
#
# Configuration is through environment variables, for example:
#
#   JULIA_NUM_THREADS=4 CLENSHAW_ORDER=1783 BENCH_SECONDS=5 \
#       julia-x86_64 --project clenshaw_unroll.jl

const ROTATE3 = SVector{3,Int}(3, 1, 2)

@inline function reset_workspace!(b)
    fill!(b[1], 0)
    fill!(b[2], 0)
    fill!(b[3], 0)
    return nothing
end

@inline function chebyshev_step!(mapping!, out, current, older, c, X)
    mapping!(out, current)
    @batch per=thread for i in eachindex(out)
        @inbounds out[i] = c * X[i] + 2 * out[i] - older[i]
    end
    return nothing
end

@inline function chebyshev_final!(mapping!, current, older, c0, Y, X)
    mapping!(Y, current)
    @batch per=thread for i in eachindex(Y)
        @inbounds Y[i] = c0 * X[i] + Y[i] - older[i]
    end
    return Y
end

"""Reference implementation: rotate the three workspace references after every step."""
function clenshaw_permute!(mapping!, coefficients, order, b, X, Y)
    reset_workspace!(b)

    @inbounds for k in order:-1:1
        chebyshev_step!(mapping!, b[1], b[2], b[3], coefficients[k + 1], X)
        permute!(b, ROTATE3)
    end

    return chebyshev_final!(mapping!, b[2], b[3], coefficients[1], Y, X)
end

"""Same recurrence using three local references instead of `permute!(b, ROTATE3)`."""
function clenshaw_localrefs!(mapping!, coefficients, order, b, X, Y)
    reset_workspace!(b)
    b1, b2, b3 = b[1], b[2], b[3]

    @inbounds for k in order:-1:1
        chebyshev_step!(mapping!, b1, b2, b3, coefficients[k + 1], X)
        b1, b2, b3 = b3, b1, b2
    end

    return chebyshev_final!(mapping!, b2, b3, coefficients[1], Y, X)
end

"""Three recurrence steps per loop body; no buffer permutation is performed."""
function clenshaw_unrolled3!(mapping!, coefficients, order, b, X, Y)
    reset_workspace!(b)
    k = order

    @inbounds while k >= 3
        # k:     b1 <- c_k X + 2 A*b2 - b3
        chebyshev_step!(mapping!, b[1], b[2], b[3], coefficients[k + 1], X)

        # k - 1: b3 <- c_{k-1} X + 2 A*b1 - b2
        chebyshev_step!(mapping!, b[3], b[1], b[2], coefficients[k], X)

        # k - 2: b2 <- c_{k-2} X + 2 A*b3 - b1
        chebyshev_step!(mapping!, b[2], b[3], b[1], coefficients[k - 1], X)

        k -= 3
    end

    # After every complete triple, the newest and older states are again b2 and b3.
    # The final partial triple determines which physical buffers hold those states.
    if k == 2
        chebyshev_step!(mapping!, b[1], b[2], b[3], coefficients[3], X)
        chebyshev_step!(mapping!, b[3], b[1], b[2], coefficients[2], X)
        return chebyshev_final!(mapping!, b[3], b[1], coefficients[1], Y, X)
    elseif k == 1
        chebyshev_step!(mapping!, b[1], b[2], b[3], coefficients[2], X)
        return chebyshev_final!(mapping!, b[1], b[2], coefficients[1], Y, X)
    else
        return chebyshev_final!(mapping!, b[2], b[3], coefficients[1], Y, X)
    end
end

function workspace_like(X)
    return [similar(X) for _ in 1:3]
end

function check_correctness!(mapping!, coefficients, order, X)
    # These exercise every possible order modulo three without making the full-order
    # recurrence run twice. The full-order comparison is made after the warm-up below.
    orders = 0:min(order, 8)

    for n in orders
        n <= order || continue

        Y_permute = similar(X)
        Y_local = similar(X)
        Y_unrolled = similar(X)

        clenshaw_permute!(
            mapping!, coefficients, n, workspace_like(X), X, Y_permute,
        )
        clenshaw_localrefs!(
            mapping!, coefficients, n, workspace_like(X), X, Y_local,
        )
        clenshaw_unrolled3!(
            mapping!, coefficients, n, workspace_like(X), X, Y_unrolled,
        )

        local_error = maximum(abs, Y_permute .- Y_local)
        unrolled_error = maximum(abs, Y_permute .- Y_unrolled)

        isapprox(Y_permute, Y_local; rtol=5e-13, atol=5e-13) ||
            error("local-reference result is wrong at order $n (max error $local_error)")
        isapprox(Y_permute, Y_unrolled; rtol=5e-13, atol=5e-13) ||
            error("three-step result is wrong at order $n (max error $unrolled_error)")

        @printf(
            "  order %-5d  local refs max error = %.3e   unrolled max error = %.3e\n",
            n,
            local_error,
            unrolled_error,
        )
    end

    return nothing
end

function rotation_only!(b, n)
    @inbounds for _ in 1:n
        permute!(b, ROTATE3)
    end
    return nothing
end

@inline median_ms(trial) = median(trial).time / 1e6

function main()
    L = parse(Int, get(ENV, "QREM_L", "17"))
    bs = parse(Int, get(ENV, "BLOCK_SIZE", "4"))
    order = parse(Int, get(ENV, "CLENSHAW_ORDER", "300"))
    bench_seconds = parse(Float64, get(ENV, "BENCH_SECONDS", "3"))
    bench_samples = parse(Int, get(ENV, "BENCH_SAMPLES", "5"))

    L >= 1 || error("QREM_L must be at least 1")
    bs >= 1 || error("BLOCK_SIZE must be at least 1")
    order >= 0 || error("CLENSHAW_ORDER must be non-negative")
    bench_seconds > 0 || error("BENCH_SECONDS must be positive")
    bench_samples >= 1 || error("BENCH_SAMPLES must be at least 1")

    BenchmarkTools.DEFAULT_PARAMETERS.seconds = bench_seconds
    BenchmarkTools.DEFAULT_PARAMETERS.samples = bench_samples
    BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

    rng = MersenneTwister(42)
    gamma = 1.0
    H = qrem_hamiltonian(L; Γ=gamma, rng=rng)
    D = size(H, 1)

    # A rigorous Gershgorin interval for QREM, used only to keep the long
    # Chebyshev recurrence numerically bounded during this benchmark.
    diagonal = diag(H)
    Emin = minimum(diagonal) - L * abs(gamma)
    Emax = maximum(diagonal) + L * abs(gamma)
    scale = (Emax - Emin) / 2
    shift = (Emax + Emin) / 2

    built = Polfed.build_optimized_mapping(
        H,
        Polfed.MappingKernel(:spin_flip, :packed; W=0, G=4),
        Val(bs),
        scale,
        shift,
    )
    mapping! = built.f!_rescaled

    X = randn(rng, bs, D)
    coefficients = randn(rng, order + 1) ./ sqrt(order + 1)

    println("Clenshaw buffer-rotation benchmark")
    println("  QREM: L=$L, D=$D, block size=$bs")
    println("  order=$order, Julia threads=$(Threads.nthreads())")
    println("  mapping kernel: ", built.kernel)
    println("  benchmark: $(bench_seconds)s maximum, $(bench_samples) samples, evals=1")

    println("\nCorrectness checks")
    check_correctness!(mapping!, coefficients, order, X)

    b_permute = workspace_like(X)
    b_local = workspace_like(X)
    b_unrolled = workspace_like(X)
    Y_permute = similar(X)
    Y_local = similar(X)
    Y_unrolled = similar(X)

    # Compile every path before benchmarking.
    clenshaw_permute!(mapping!, coefficients, order, b_permute, X, Y_permute)
    clenshaw_localrefs!(mapping!, coefficients, order, b_local, X, Y_local)
    clenshaw_unrolled3!(mapping!, coefficients, order, b_unrolled, X, Y_unrolled)

    full_local_error = maximum(abs, Y_permute .- Y_local)
    full_unrolled_error = maximum(abs, Y_permute .- Y_unrolled)
    isapprox(Y_permute, Y_local; rtol=5e-13, atol=5e-13) ||
        error("local-reference result is wrong at full order $order (max error $full_local_error)")
    isapprox(Y_permute, Y_unrolled; rtol=5e-13, atol=5e-13) ||
        error("three-step result is wrong at full order $order (max error $full_unrolled_error)")
    @printf("  full order %-5d  local refs max error = %.3e   unrolled max error = %.3e\n",
            order, full_local_error, full_unrolled_error)

    println("\nWhole Clenshaw application")
    permute_trial = @benchmark clenshaw_permute!(
        $mapping!, $coefficients, $order, $b_permute, $X, $Y_permute,
    ) evals=1
    local_trial = @benchmark clenshaw_localrefs!(
        $mapping!, $coefficients, $order, $b_local, $X, $Y_local,
    ) evals=1
    unrolled_trial = @benchmark clenshaw_unrolled3!(
        $mapping!, $coefficients, $order, $b_unrolled, $X, $Y_unrolled,
    ) evals=1

    permute_ms = median_ms(permute_trial)
    local_ms = median_ms(local_trial)
    unrolled_ms = median_ms(unrolled_trial)

    @printf("  permute! each step : %10.3f ms\n", permute_ms)
    @printf("  local references   : %10.3f ms  (%+.3f%% vs permute!)\n",
            local_ms, 100 * (local_ms / permute_ms - 1))
    @printf("  three-step unrolled: %10.3f ms  (%+.3f%% vs permute!)\n",
            unrolled_ms, 100 * (unrolled_ms / permute_ms - 1))

    small_buffers = [zeros(1), zeros(1), zeros(1)]
    rotation_trial = @benchmark rotation_only!($small_buffers, $order) evals=1
    rotation_ns = median(rotation_trial).time

    println("\nCost of permuting only the three references")
    @printf("  all %d permutations: %.3f microseconds\n", order, rotation_ns / 1e3)
    order > 0 && @printf("  per permutation:     %.3f nanoseconds\n", rotation_ns / order)

    saved_percent = 100 * (permute_ms - unrolled_ms) / permute_ms
    println("\nConclusion for this run")
    if abs(saved_percent) < 1
        @printf("  No meaningful difference: unrolling changed runtime by %+.3f%%.\n",
                -saved_percent)
    elseif saved_percent > 0
        @printf("  Three-step unrolling was %.3f%% faster. Repeat with a longer BENCH_SECONDS before adopting it.\n",
                saved_percent)
    else
        @printf("  Three-step unrolling was %.3f%% slower.\n", -saved_percent)
    end

    return nothing
end

main()
