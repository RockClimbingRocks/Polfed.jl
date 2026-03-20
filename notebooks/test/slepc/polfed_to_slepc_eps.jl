#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Random
using SparseArrays

include("/home/rokpintar/projects/Polfed/src/Polfed.jl")
using .Polfed

"""
Construct XXZ Hamiltonian in fixed-magnetization sector (same model used in repo docs/tests).
"""
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

"""
Normalized random vector / orthonormal block.
"""
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

"""
Exact bounds for small/medium test matrices so POLFED and shell mapping use the same rescaling.
"""
function exact_bounds(H::AbstractMatrix{<:Real})
    vals = eigvals(Hermitian(Matrix(H)))
    return first(vals), last(vals)
end

"""
Create plain and rescaled mappings matching POLFED's internal rescaling:
  H̃ = (H - b I) / a, where a=(Emax-Emin)/2 and b=(Emax+Emin)/2.
"""
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

"""
Build the same Chebyshev filter mapping that POLFED used, with order/target read from report.
"""
function build_report_filter_mapping(
    Hrescaled!::Function,
    dim::Int;
    target_rescaled::Float64,
    order::Int,
    normalization::Float64=1.0,
    polynomialtype::String="Chebyshev",
)
    polynomialtype == "Chebyshev" || error("This notebook test currently supports report polynomialtype=Chebyshev only.")

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

const HAVE_SLEPC = let
    ok = true
    try
        @eval using MPI
        @eval using PetscWrap
        @eval using SlepcWrap
    catch err
        ok = false
        @warn "Skipping SLEPc run: MPI/PetscWrap/SlepcWrap not available in this environment." exception=(err, catch_backtrace())
    end
    ok
end

Base.@kwdef struct EPSConfig
    nev::Int = 20
    ncv::Int = 80
    mpd::Int = 0
    selection::Symbol = :target_real  # :target_real | :largest_real | :smallest_real
    tol::Float64 = 1e-10
    maxit::Int = 10_000
    target::Union{Nothing, Float64} = nothing
end

function eps_options(cfg::EPSConfig)
    opt = String[
        "-mat_type", "shell",
        "-eps_type", "krylovschur",
        "-eps_nev", string(cfg.nev),
        "-eps_ncv", string(cfg.ncv),
        "-eps_tol", string(cfg.tol),
        "-eps_max_it", string(cfg.maxit),
    ]

    if cfg.mpd > 0
        push!(opt, "-eps_mpd")
        push!(opt, string(cfg.mpd))
    end

    if cfg.selection === :target_real
        push!(opt, "-eps_target_real")
    elseif cfg.selection === :largest_real
        push!(opt, "-eps_largest_real")
    elseif cfg.selection === :smallest_real
        push!(opt, "-eps_smallest_real")
    else
        error("Unsupported selection=$(cfg.selection).")
    end

    if cfg.target !== nothing
        push!(opt, "-eps_target")
        push!(opt, string(cfg.target))
    end

    return opt
end

function vec_get_array_read(X)
    if isdefined(PetscWrap, :VecGetArrayRead)
        x_local = PetscWrap.VecGetArrayRead(X)
        return x_local, :read
    elseif isdefined(PetscWrap, :VecGetArray)
        x_local = PetscWrap.VecGetArray(X)
        return x_local, :rw
    end
    return PetscWrap.vec2array(X), :copy
end

function vec_restore_array_read(X, x_local, mode::Symbol)
    if mode === :read
        PetscWrap.VecRestoreArrayRead(X, x_local)
    elseif mode === :rw
        PetscWrap.VecRestoreArray(X, x_local)
    end
    return nothing
end

function vec_get_array_write(Y)
    if isdefined(PetscWrap, :VecGetArray)
        y_local = PetscWrap.VecGetArray(Y)
        return y_local, :rw
    end
    return PetscWrap.vec2array(Y), :copy
end

function vec_restore_array_write(Y, y_local, mode::Symbol)
    if mode === :rw
        PetscWrap.VecRestoreArray(Y, y_local)
    else
        for i in eachindex(y_local)
            Y[i] = y_local[i]
        end
    end
    return nothing
end

function register_shell_mul!(A, matmult!::Function)
    if isdefined(PetscWrap, :set_shell_mul!)
        return PetscWrap.set_shell_mul!(A, matmult!)
    end

    if isdefined(PetscWrap, :MatShellSetOperation) && isdefined(PetscWrap, :MATOP_MULT)
        PetscWrap.MatShellSetOperation(A, PetscWrap.MATOP_MULT, matmult!)
        return matmult!
    end

    if @isdefined(MatShellSetOperation) && @isdefined(MATOP_MULT)
        MatShellSetOperation(A, MATOP_MULT, matmult!)
        return matmult!
    end

    error("Unable to register shell MatMult callback. Expected `set_shell_mul!` or `MatShellSetOperation` in PetscWrap.")
end

"""
Shell EPS solve for a local mapping y := A*x.
This notebook test is intentionally rank-local; run with a single MPI rank.
"""
function solve_eps_shell(N::Int, Hmul!::Function; cfg::EPSConfig=EPSConfig(), ctx=nothing)
    HAVE_SLEPC || error("SLEPc stack is not available. Install MPI, PetscWrap, and SlepcWrap first.")
    t_total_start = time_ns()

    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    nprocs == 1 || error("This shell test expects one MPI rank (`mpirun -n 1 ...`).")

    SlepcInitialize(eps_options(cfg))

    A = MatCreate()
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N)
    MatSetFromOptions(A)
    MatSetUp(A)

    shell_ctx = (ctx = ctx,)

    function matmult!(Y, X)
        x_local, x_mode = vec_get_array_read(X)
        y_local, y_mode = vec_get_array_write(Y)

        try
            Hmul!(y_local, x_local, shell_ctx.ctx)
        finally
            vec_restore_array_read(X, x_local, x_mode)
            vec_restore_array_write(Y, y_local, y_mode)
        end

        return nothing
    end

    shell_mul_handle = register_shell_mul!(A, matmult!)

    eps = EPSCreate()
    EPSSetOperators(eps, A)
    if isdefined(SlepcWrap, :EPS_HEP)
        EPSSetProblemType(eps, SlepcWrap.EPS_HEP)
    end
    EPSSetFromOptions(eps)
    EPSSetUp(eps)
    eps_solve_seconds = @elapsed EPSSolve(eps)

    nconv = EPSGetConverged(eps)
    evals = Vector{ComplexF64}(undef, nconv)
    for i in 0:nconv-1
        er, ei = EPSGetEigenvalue(eps, i)
        evals[i + 1] = complex(er, ei)
    end

    EPSDestroy(eps)
    MatDestroy(A)

    shell_mul_handle = nothing
    GC.gc()
    SlepcFinalize()

    total_seconds = (time_ns() - t_total_start) / 1e9
    rank == 0 && @printf("SLEPc converged eigenpairs: %d\n", nconv)
    stats = (
        nconv = nconv,
        eps_solve_seconds = eps_solve_seconds,
        total_seconds = total_seconds,
    )
    return evals, stats
end

function run_demo(; L::Int=12, delta::Float64=1.0, howmany::Int=30, target::Float64=0.0, blocksize::Int=1)
    H = construct_xxz_spin_sector(L, delta, L ÷ 2)
    dim = size(H, 1)
    x0 = make_x0(dim, blocksize)

    Emin, Emax = exact_bounds(H)

    mapping = Polfed.MappingConfig(
        parallel_strategy = Polfed.NoParallel(),
        Emin = Emin,
        Emax = Emax,
    )
    transform = Polfed.TransformConfig()

    fact = Polfed.FactorizationConfig(overestimate_iters = 2.0)

    local vals
    local vecs
    local report
    polfed_seconds = @elapsed begin
        vals, vecs, report = Polfed.polfed(
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
    @printf("\nTiming: POLFED call walltime = %.6f s\n", polfed_seconds)

    println("\n=== POLFED REPORT (from structured object) ===")
    Polfed.display_report(report; use_colors=false)

    st = report.spectral_transform
    target_rescaled = Float64(st.target)
    target_unrescaled = ((Emax - Emin) / 2) * target_rescaled + ((Emax + Emin) / 2)

    println("\n=== EXTRACTED REPORT DATA ===")
    @printf("order K                = %d\n", st.order)
    @printf("target (rescaled)      = %.10f\n", st.target)
    @printf("target (unrescaled)    = %.10f\n", target_unrescaled)
    @printf("interval [left,right]  = [%.10f, %.10f]\n", st.left, st.right)
    @printf("order safety factor    = %.6f\n", st.order_safety_factor)
    @printf("polynomial type        = %s\n", st.polynomialtype)

    Hmul!, Hrescaled!, _, _ = build_base_mappings(H, Emin, Emax)
    filtered_mul! = build_report_filter_mapping(
        Hrescaled!,
        dim;
        target_rescaled = target_rescaled,
        order = st.order,
        normalization = transform.normalization,
        polynomialtype = st.polynomialtype,
    )

    if !HAVE_SLEPC
        println("\nSLEPc part skipped (missing MPI/PetscWrap/SlepcWrap).")
        return vals, vecs, report, nothing, nothing
    end

    nev = min(howmany, dim)
    ncv = max(2 * nev, 40)

    println("\n=== SLEPc EPS ON ORIGINAL MAPPING (H*x) ===")
    local evals_h
    local eps_h_stats
    evals_h, eps_h_stats = solve_eps_shell(
        dim,
        Hmul!;
        cfg = EPSConfig(
            nev = nev,
            ncv = ncv,
            selection = :target_real,
            target = target_unrescaled,
        ),
    )
    @printf("Timing: EPS(H) solve stage = %.6f s, EPS(H) total call = %.6f s\n", eps_h_stats.eps_solve_seconds, eps_h_stats.total_seconds)
    println("First few EPS(H) eigenvalues: ", real.(evals_h[1:min(end, 5)]))

    println("\n=== SLEPc EPS ON REPORT-MATCHED FILTER MAPPING (P_K(H_tilde)*x) ===")
    local evals_filtered
    local eps_f_stats
    evals_filtered, eps_f_stats = solve_eps_shell(
        dim,
        filtered_mul!;
        cfg = EPSConfig(
            nev = nev,
            ncv = ncv,
            selection = :largest_real,
            target = nothing,
        ),
    )
    @printf("Timing: EPS(filtered) solve stage = %.6f s, EPS(filtered) total call = %.6f s\n", eps_f_stats.eps_solve_seconds, eps_f_stats.total_seconds)
    println("First few EPS(filtered) eigenvalues: ", real.(evals_filtered[1:min(end, 5)]))

    polfed_vals = sort(real.(collect(vals)); by = x -> abs(x - target_unrescaled))
    slepc_vals = sort(real.(evals_h); by = x -> abs(x - target_unrescaled))
    ncmp = min(length(polfed_vals), length(slepc_vals), 10)
    if ncmp > 0
        max_abs_diff = maximum(abs.(polfed_vals[1:ncmp] .- slepc_vals[1:ncmp]))
        @printf("\nMax |POLFED - SLEPc(H)| over first %d target-nearest eigenvalues: %.3e\n", ncmp, max_abs_diff)
    end

    return vals, vecs, report, evals_h, evals_filtered
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

    println("Running POLFED->report->SLEPc notebook test with:")
    @printf("L=%d, delta=%.3f, howmany=%d, target=%.6f, blocksize=%d\n", L, delta, howmany, target, blocksize)

    run_demo(
        L = L,
        delta = delta,
        howmany = howmany,
        target = target,
        blocksize = blocksize,
    )
end
