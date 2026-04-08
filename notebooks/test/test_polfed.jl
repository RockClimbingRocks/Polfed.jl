using Pkg

const ROOT_DIR = normpath(joinpath(@__DIR__, "..", ".."))

Pkg.activate(ROOT_DIR)

using LinearAlgebra
using Random
using SparseArrays

include(joinpath(ROOT_DIR, "src", "Polfed.jl"))
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

            szsz = (0.5 - si) * (0.5 - sj)
            push!(rows, col)
            push!(cols, col)
            push!(vals, delta * szsz)

            if si != sj
                flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                if haskey(bmap, flipped)
                    push!(rows, bmap[flipped])
                    push!(cols, col)
                    push!(vals, 0.5)
                end
            end
        end
    end

    return sparse(rows, cols, vals, dim, dim)
end

function nearest_eigs(vals::AbstractVector, target::Real, howmany::Int)
    perm = sortperm(eachindex(vals); by = i -> (abs(vals[i] - target), vals[i]))
    return vals[perm[1:howmany]]
end

function main()
    Random.seed!(1234)

    L = 10
    Nup = L ÷ 2
    delta = 0.7
    target = 0.0
    howmany = 4

    mat = construct_xxz_spin_sector(L, delta, Nup)
    dim = size(mat, 1)

    v0 = randn(dim)
    v0 ./= norm(v0)

    println("Running Polfed smoke test")
    println("L = $L, Nup = $Nup, dim = $dim, delta = $delta, target = $target, howmany = $howmany")

    vals, vecs, report = Polfed.polfed(mat, v0, howmany, target; produce_report = true)

    nfound = length(vals)
    nfound == 0 && error("Polfed returned no eigenvalues.")

    exact_vals = eigvals(Matrix(mat))
    vals_sorted = sort(collect(vals))
    exact_sorted = sort(collect(nearest_eigs(exact_vals, target, nfound)))
    max_eval_err = maximum(abs.(vals_sorted .- exact_sorted))
    residuals = [norm(mat * vecs[:, i] - vals[i] * vecs[:, i]) for i in axes(vecs, 2)]
    max_residual = maximum(residuals)

    println("Computed eigenvalues near target:")
    display(vals_sorted)
    println("Reference eigenvalues near target:")
    display(exact_sorted)
    println("Max eigenvalue error: $max_eval_err")
    println("Max residual norm: $max_residual")

    Polfed.display_report(report)

    atol_eval = 1e-8
    atol_residual = 1e-8

    if max_eval_err > atol_eval
        error("Eigenvalue check failed: max error $max_eval_err > $atol_eval")
    end

    if max_residual > atol_residual
        error("Residual check failed: max residual $max_residual > $atol_residual")
    end

    println("Polfed smoke test passed.")
end

main()
