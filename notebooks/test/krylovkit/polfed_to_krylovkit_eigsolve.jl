#!/usr/bin/env julia

using KrylovKit
using LinearAlgebra
using Printf
using Random
using SparseArrays

include(joinpath(@__DIR__, "..", "..", "..", "src", "Polfed.jl"))
using .Polfed

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

function make_x0(dim::Int, blocksize::Int; seed::Int=1234)
    rng = MersenneTwister(seed)
    if blocksize == 1
        x0 = randn(rng, dim)
        x0 ./= norm(x0)
        return x0
    end
    x0_ = randn(rng, dim, blocksize)
    return Matrix(qr(x0_).Q)
end

function exact_bounds(H::AbstractMatrix{<:Real})
    vals = eigvals(Hermitian(Matrix(H)))
    return first(vals), last(vals)
end

function build_base_mappings(H::SparseMatrixCSC{Float64,Int}, Emin::Real, Emax::Real)
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    Hmul! = function (y::AbstractVector, x::AbstractVector, _ctx)
        mul!(y, H, x)
        return nothing
    end

    Hrescaled! = function (y::AbstractVector, x::AbstractVector, _ctx)
        mul!(y, H, x)
        @. y = (y - b * x) / a
        return nothing
    end

    return Hmul!, Hrescaled!, a, b
end

function build_report_filter_mapping(
    Hrescaled!::Function,
    dim::Int;
    target_rescaled::Float64,
    order::Int,
    normalization::Float64=1.0,
    polynomialtype::String="Chebyshev",
)
    polynomialtype == "Chebyshev" || error("Only Chebyshev report polynomial type is supported in this script.")

    coefficients(λ::T, n::Int) where {T<:Real} = T((2 - ==(n, 0)) * cos(n * acos(λ)))
    Hrescaled2! = (Y, X) -> Hrescaled!(Y, X, nothing)

    transform = Polfed.ClenshawMapping.Clenshaw(
        :Chebyshev,
        n -> coefficients(target_rescaled, n),
        order,
        Hrescaled2!,
        dim,
        Float64,
    )

    norm_ = normalization / transform(target_rescaled)
    coefficients_normalized(n::Int) = coefficients(target_rescaled, n) * norm_

    clenshaw = Polfed.ClenshawMapping.Clenshaw(
        :Chebyshev,
        coefficients_normalized,
        order,
        Hrescaled2!,
        dim,
        Float64,
    )

    b_storage = [zeros(Float64, dim) for _ in 1:3]

    filtered_mul! = function (y::AbstractVector, x::AbstractVector, _ctx)
        clenshaw(y, x, b_storage)
        return nothing
    end

    return filtered_mul!
end

function run_krylovkit(
    filtered_mul!::Function,
    Hmul!::Function,
    xstart::AbstractVector,
    howmany::Int;
    tol::Float64=1e-9,
    krylovdim::Int=min(length(xstart), max(8 * howmany, 120)),
    maxiter::Int=1,
    eager::Bool=false,
    verbosity::Int=KrylovKit.WARN_LEVEL,
)
    op = function (x::AbstractVector)
        y = similar(x)
        filtered_mul!(y, x, nothing)
        return y
    end

    local vals_filter
    local vecs_filter
    local info
    alg = KrylovKit.Lanczos(
        krylovdim = krylovdim,
        maxiter = maxiter,
        tol = tol,
        eager = eager,
        verbosity = verbosity,
    )
    krylovkit_seconds = @elapsed begin
        vals_filter, vecs_filter, info = KrylovKit.eigsolve(op, xstart, howmany, :LM, alg)
    end

    Hv = similar(xstart)
    vals_from_H = Float64[]
    for v in vecs_filter
        Hmul!(Hv, v, nothing)
        λ = real(dot(v, Hv) / dot(v, v))
        push!(vals_from_H, λ)
    end

    return vals_filter, vecs_filter, vals_from_H, info, krylovkit_seconds, alg
end

function run_demo(;
    L::Int=12,
    delta::Float64=1.0,
    howmany::Int=30,
    target::Float64=0.0,
    blocksize::Int=1,
    kk_krylovdim::Int=0,
    kk_maxiter::Int=1,
    kk_tol::Float64=1e-9,
)
    H = construct_xxz_spin_sector(L, delta, L ÷ 2)
    dim = size(H, 1)
    x0 = make_x0(dim, blocksize)
    xstart = x0 isa AbstractMatrix ? copy(view(x0, :, 1)) : copy(x0)

    Emin, Emax = exact_bounds(H)

    mapping = Polfed.MappingConfig(
        parallel_strategy = Polfed.NoParallel(),
        Emin = Emin,
        Emax = Emax,
    )
    transform = Polfed.TransformConfig()
    fact = Polfed.FactorizationConfig(overestimate_iters=2.0)

    local vals_polfed
    local vecs_polfed
    local report
    polfed_seconds = @elapsed begin
        vals_polfed, vecs_polfed, report = Polfed.polfed(
            H,
            x0,
            howmany,
            target;
            produce_report = true,
            mapping = mapping,
            transform = transform,
            fact = fact,
        )
    end
    @printf("\nTiming: POLFED walltime = %.6f s\n", polfed_seconds)
    println("\n=== POLFED REPORT ===")
    Polfed.display_report(report; use_colors=false)

    st = report.spectral_transform
    target_rescaled = Float64(st.target)
    target_unrescaled = ((Emax - Emin) / 2) * target_rescaled + ((Emax + Emin) / 2)

    println("Report-derived mapping parameters:")
    @printf("- order K              = %d\n", st.order)
    @printf("- target (rescaled)    = %.10f\n", st.target)
    @printf("- target (unrescaled)  = %.10f\n", target_unrescaled)
    @printf("- interval [l, r]      = [%.10f, %.10f]\n", st.left, st.right)

    Hmul!, Hrescaled!, _, _ = build_base_mappings(H, Emin, Emax)
    filtered_mul! = build_report_filter_mapping(
        Hrescaled!,
        dim;
        target_rescaled = target_rescaled,
        order = st.order,
        normalization = transform.normalization,
        polynomialtype = st.polynomialtype,
    )

    kk_kdim = kk_krylovdim > 0 ? kk_krylovdim : min(dim, max(8 * howmany, 120))

    vals_filter, vecs_filter, vals_krylov_h, info, krylovkit_seconds, kk_alg = run_krylovkit(
        filtered_mul!,
        Hmul!,
        xstart,
        howmany;
        tol = kk_tol,
        krylovdim = kk_kdim,
        maxiter = kk_maxiter,
        eager = false,
        verbosity = KrylovKit.WARN_LEVEL,
    )
    println("KrylovKit settings:")
    @printf("- algorithm            = Lanczos\n")
    @printf("- krylovdim            = %d\n", kk_alg.krylovdim)
    @printf("- maxiter              = %d (1 means no restart)\n", kk_alg.maxiter)
    @printf("- tol                  = %.2e\n", kk_alg.tol)
    @printf("Timing: KrylovKit eigsolve walltime = %.6f s\n", krylovkit_seconds)

    println("\nKrylovKit convergence summary:")
    println("- converged: ", info.converged)
    println("- numiter:   ", info.numiter)
    println("- numops:    ", info.numops)

    pvals = sort(real.(collect(vals_polfed)); by = x -> abs(x - target_unrescaled))
    kvals = sort(vals_krylov_h; by = x -> abs(x - target_unrescaled))

    ncmp = min(10, length(pvals), length(kvals))
    if ncmp > 0
        max_abs_diff = maximum(abs.(pvals[1:ncmp] .- kvals[1:ncmp]))
        @printf("Max |POLFED - KrylovKit| over first %d near-target values: %.3e\n", ncmp, max_abs_diff)
    end

    return vals_polfed, vecs_polfed, report, vals_filter, vecs_filter, vals_krylov_h, info
end

function parse_arg(args::Vector{String}, i::Int, default, ::Type{T}) where {T}
    return length(args) >= i ? parse(T, args[i]) : default
end

if abspath(PROGRAM_FILE) == @__FILE__
    L = parse_arg(ARGS, 1, 12, Int)
    delta = parse_arg(ARGS, 2, 1.0, Float64)
    howmany = parse_arg(ARGS, 3, 30, Int)
    target = parse_arg(ARGS, 4, 0.0, Float64)
    blocksize = parse_arg(ARGS, 5, 1, Int)
    kk_krylovdim = parse_arg(ARGS, 6, 0, Int)      # 0 => auto
    kk_maxiter = parse_arg(ARGS, 7, 1, Int)        # 1 => no restart
    kk_tol = parse_arg(ARGS, 8, 1e-9, Float64)

    println("Running POLFED -> report-matched mapping -> KrylovKit.eigsolve")
    @printf("L=%d, delta=%.3f, howmany=%d, target=%.6f, blocksize=%d, kk_krylovdim=%d, kk_maxiter=%d, kk_tol=%.2e\n",
        L, delta, howmany, target, blocksize, kk_krylovdim, kk_maxiter, kk_tol)

    run_demo(
        L = L,
        delta = delta,
        howmany = howmany,
        target = target,
        blocksize = blocksize,
        kk_krylovdim = kk_krylovdim,
        kk_maxiter = kk_maxiter,
        kk_tol = kk_tol,
    )
end
