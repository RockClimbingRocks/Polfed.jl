using SparseArrays, LinearAlgebra, BenchmarkTools, Random

# Load Polfed from the local checkout so benchmarks always use the current code.
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
using .Polfed
const PolfedCore = Polfed.PolfedCore


function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
    basis = [b for b in 0:2^L-1 if count_ones(b) == Nup] # generate basis
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))  # state index map
    rows, cols, vals = Int[], Int[], Float64[]
    for (col, state) in enumerate(basis)
        for i in 1:L
            j = i % L + 1  # PBC
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1
            SzSz = (0.5 - si) * (0.5 - sj)
            push!(rows, col); push!(cols, col); push!(vals, delta * SzSz)
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

function extract_sparse_data(A::SparseMatrixCSC)
    dim = size(A, 1)

    diags = Vector(diag(A))

    offdiags_flatten = Int[]
    sizehint!(offdiags_flatten, nnz(A) - dim)
    start_indices = zeros(Int, dim)

    for j in 1:dim
        start_indices[j] = length(offdiags_flatten) + 1
        for ptr in A.colptr[j]:(A.colptr[j+1] - 1)
            i = A.rowval[ptr]
            i != j && push!(offdiags_flatten, i)
        end
    end

    val = A[1, offdiags_flatten[1]]
    return diags, offdiags_flatten, start_indices, val
end

function get_diags_and_offdiagonals_by_value(mat::AbstractMatrix{T}; tol=1e-14, round_digits=15) where {T<:Real}
    dim = size(mat, 1)

    value_to_conn = Dict{Float64, Vector{Vector{Int}}}()
    diagonals = Vector{Float64}(undef, dim)

    for i in 1:dim
        diagonals[i] = round(mat[i, i]; digits=round_digits)

        row_conns = Dict{Float64, Vector{Int}}()
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            i == j && continue
            v = mat[i, j]
            abs(v) < tol && continue
            v_rounded = round(v; digits=round_digits)
            push!(get!(row_conns, v_rounded, Int[]), j)
        end
        for (v, js) in row_conns
            if !haskey(value_to_conn, v)
                value_to_conn[v] = [Int[] for _ in 1:dim]
            end
            value_to_conn[v][i] = js
        end
    end

    offdiagonals = Tuple{Float64, Vector{Int}, Vector{Int}}[]
    for (v, conn_lists) in sort(collect(value_to_conn), by=first)
        flat = Int[]
        starts = Int[]
        idx = 1
        for js in conn_lists
            push!(starts, idx)
            append!(flat, js)
            idx += length(js)
        end
        push!(offdiagonals, (v, flat, starts))
    end

    return diagonals, offdiagonals
end

function mapvec_with_xxz!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    return (Y, X) -> begin
        J_half = J / 2
        for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
        end
    end
end

function mapvec_with_xxz_parallel!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    return (Y, X) -> begin
        J_half = J / 2
        @threads for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], J_half * sum_val)
        end
    end
end

function clenshaw_with_xxz!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2

    crr = @inline (b1::AbstractVector, b2::AbstractVector, b3::AbstractVector, c::Real, X::AbstractVector) -> begin
        for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b2[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b2[i], J_half * sum_val)
            @inbounds b1[i] = c * X[i] + 2 * yi - b3[i]
        end
    end

    cfs = @inline (b1::AbstractVector, b2::AbstractVector, c::Real, Y::AbstractVector, X::AbstractVector) -> begin
        for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b1[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b1[i], J_half * sum_val)
            @inbounds Y[i] = c * X[i] + yi - b2[i]
        end
    end

    return crr, cfs
end

function clenshaw_with_xxz_parallel!(
    diags::Vector{Float64},
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
    J::Float64
)
    J_half = J / 2

    crr = @inline (b1::AbstractVector, b2::AbstractVector, b3::AbstractVector, c::Real, X::AbstractVector) -> begin
        @threads for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b2[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b2[i], J_half * sum_val)
            @inbounds b1[i] = c * X[i] + 2 * yi - b3[i]
        end
    end

    cfs = @inline (b1::AbstractVector, b2::AbstractVector, c::Real, Y::AbstractVector, X::AbstractVector) -> begin
        @threads for i in eachindex(start_indices)
            @inbounds start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += b1[offdiags_flatten[j]]
            end

            @inbounds yi = muladd(diags[i], b1[i], J_half * sum_val)
            @inbounds Y[i] = c * X[i] + yi - b2[i]
        end
    end

    return crr, cfs
end


function run_case(label::AbstractString, f::Function)
    println("\n== ", label, " ==")
    GC.gc()
    f() # warm-up to compile
    GC.gc()
    trial = @benchmark $f() samples=3 evals=1
    show(stdout, MIME("text/plain"), trial)
    println()
    return trial
end

run_case(f::Function, label::AbstractString) = run_case(label, f)


function main()
    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
    howmany = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    target = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0

    delta = 1.234
    Nup = L ÷ 2

    println("Building XXZ model...")
    mat = construct_xxz_spin_sector(L, delta, Nup)
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

    diags, offdiags_flatten, start_indices, val = extract_sparse_data(mat)
    J = 2 * val
    use_parallel = Threads.nthreads() > 1
    map_H = use_parallel ?
        mapvec_with_xxz_parallel!(diags, offdiags_flatten, start_indices, J) :
        mapvec_with_xxz!(diags, offdiags_flatten, start_indices, J)

    crr_H, cfs_H = use_parallel ?
        clenshaw_with_xxz_parallel!(diags, offdiags_flatten, start_indices, J) :
        clenshaw_with_xxz!(diags, offdiags_flatten, start_indices, J)

    v0 = rand(dim)
    v0 ./= norm(v0)
    Emin = first(collect(Polfed.Lanczos.lanczos(map_H, v0, 1; which=:SR, maxdim=1000)[1]))
    Emax = last(collect(Polfed.Lanczos.lanczos(map_H, v0, 1; which=:LR,  maxdim=1000)[1]))
    a = (Emax - Emin) / 2
    b = (Emax + Emin) / 2

    f!_rescaled = (Y, X) -> begin
        map_H(Y, X)
        @. Y = (Y - b * X) / a
    end

    crr_rescaled = (b1, b2, b3, c, X) -> begin
        crr_H(b1, b2, b3, c, X)
        @. b1 = c * X + 2 * ((b1 - c * X + b3) / 2 - b * b2) / a - b3
    end

    cfs_rescaled = (b1, b2, c, Y, X) -> begin
        cfs_H(b1, b2, c, Y, X)
        @. Y = c * X + ((Y - c * X + b2) - b * b1) / a - b2
    end

    # Case 3: provide optimized rescaled mapping via MappingConfig
    mapping_rescaled = MappingConfig(
        parallel_strategy=MulColsParallel(),
        f!_rescaled=f!_rescaled,
        Emin=Emin,
        Emax=Emax,
    )

    run_case("Optimized rescaled mapping (MappingConfig)") do
        polfed(map_H, x0, howmany, target;
            mapping=mapping_rescaled,
            produce_report=false
        )
    end

    # Case 4: provide optimized rescaled mapping + Clenshaw kernels
    mapping_clenshaw = MappingConfig(
        parallel_strategy=MulColsParallel(),
        f!_rescaled=f!_rescaled,
        clenshaw_recurrence=crr_rescaled,
        clenshaw_finalsum=cfs_rescaled,
        Emin=Emin,
        Emax=Emax,
    )

    run_case("Optimized rescaled mapping + Clenshaw kernels") do
        polfed(map_H, x0, howmany, target;
            mapping=mapping_clenshaw,
            produce_report=false
        )
    end
end

main()
