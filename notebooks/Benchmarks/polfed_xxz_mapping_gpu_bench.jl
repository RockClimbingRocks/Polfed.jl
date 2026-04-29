using SparseArrays
using LinearAlgebra
using Random
using Statistics
using CUDA

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed


function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
    basis = [b for b in 0:2^L-1 if count_ones(b) == Nup]
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))
    rows, cols, vals = Int[], Int[], Float64[]

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


function summarize_times(label::AbstractString, times::Vector{Float64})
    println(label)
    println("  runs (s): ", join(string.(round.(times; digits=6)), ", "))
    println("  min/median/mean (s): ",
        round(minimum(times); digits=6), " / ",
        round(median(times); digits=6), " / ",
        round(mean(times); digits=6))
end


function time_case!(f::Function; reps::Int=8)
    f()
    CUDA.synchronize()

    times = Float64[]
    for _ in 1:reps
        GC.gc()
        t = @elapsed begin
            f()
            CUDA.synchronize()
        end
        push!(times, t)
    end

    return times
end


function main()
    CUDA.functional() || error("CUDA is not functional on this machine.")

    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    reps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    delta = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
    smoke_howmany = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
    target = 0.0
    nup = L ÷ 2

    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("Building XXZ matrix for L=", L, ", Nup=", nup, ", delta=", delta)
    mat_cpu = construct_xxz_spin_sector(L, delta, nup)
    dim = size(mat_cpu, 1)
    println("Hilbert space dimension: ", dim)
    println("nnz: ", nnz(mat_cpu))

    mat_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(mat_cpu)

    Random.seed!(1)
    x_cpu = rand(Float64, dim)
    x_cpu ./= norm(x_cpu)
    x_gpu = CuArray(x_cpu)
    y_base = similar(x_gpu)
    y_opt = similar(x_gpu)

    diags_host, offdiagonals_host = Polfed.PolfedCore.get_diags_and_offdiagonals_by_value(mat_cpu)
    diags_gpu, offdiagonals_gpu = Polfed.PolfedCore.move_packed_mapping_to_gpu(diags_host, offdiagonals_host)
    opt_map! = Polfed.optimized_mapping!(diags_gpu, offdiagonals_gpu, Polfed.NoParallel())

    CUDA.@sync mul!(y_base, mat_gpu, x_gpu)
    CUDA.@sync opt_map!(y_opt, x_gpu)
    relerr = norm(Array(y_opt .- y_base)) / max(norm(Array(y_base)), eps(Float64))
    println("Mapping relative error: ", relerr)

    if !isfinite(relerr) || relerr > 1e-10
        error("Optimized GPU mapping does not match regular GPU matrix mapping.")
    end

    println("\nSmoke test: end-to-end POLFED on GPU with optimize_mapping=true")
    x0_smoke = copy(x_gpu)
    vals_opt, vecs_opt, report_opt = Polfed.polfed(
        mat_gpu,
        x0_smoke,
        smoke_howmany,
        target;
        produce_report=true,
        mapping=Polfed.MappingConfig(optimize_mapping=true),
    )
    vals_head = collect(vals_opt)
    println("Smoke test eigenvalues (first up to 5): ", vals_head[1:min(length(vals_head), 5)])
    Polfed.display_report(report_opt)

    println("\nBenchmarking mapping-only kernels on GPU")
    baseline_times = time_case!((() -> begin
        mul!(y_base, mat_gpu, x_gpu)
    end); reps=reps)
    optimized_times = time_case!((() -> begin
        opt_map!(y_opt, x_gpu)
    end); reps=reps)

    summarize_times("Regular GPU matrix mapping (`mul!` on CuSparse)", baseline_times)
    summarize_times("Optimized GPU grouped mapping", optimized_times)
    println("Speedup (median baseline / median optimized): ", round(median(baseline_times) / median(optimized_times); digits=3), "x")
end


main()
