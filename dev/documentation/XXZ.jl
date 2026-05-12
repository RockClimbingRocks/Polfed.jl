using Polfed.Models: xxz_hamiltonian
using SparseArrays
using LinearAlgebra

construct_XXZ_matrix(L::Int, Delta::Real, Lup::Int) =
    xxz_hamiltonian(
        L,
        Lup,
        1.0,
        Delta,
        0.0;
        boundary=:periodic,
        field=0.0,
        use_sparse=true,
    )

function get_diags_and_offdiagonals_single_value(Delta::Real, L::Int, Lup::Int; tol=1e-13, round_digits=14)
    mat = construct_XXZ_matrix(L, Delta, Lup)
    dim = size(mat, 1)
    diagonals = [round(mat[i, i]; digits=round_digits) for i in 1:dim]
    flat = Int[]
    starts = Int[]
    idx = 1
    offdiag_val::Union{Nothing, Float64} = nothing

    for i in 1:dim
        push!(starts, idx)
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            i == j && continue
            v = mat[i, j]
            abs(v) < tol && continue

            v_rounded = round(v; digits=round_digits)
            if offdiag_val === nothing
                offdiag_val = v_rounded
            elseif abs(v_rounded - offdiag_val) > tol
                error("Matrix has multiple distinct off-diagonal values (found $v_rounded and $offdiag_val).")
            end
            push!(flat, j)
        end
        idx += length(flat) - starts[end] + 1
    end

    offdiag_val === nothing && error("No off-diagonal elements found above tolerance.")
    return diagonals, offdiag_val, flat, starts
end

function mapvec_with_xxz!(
    diags::Vector{Float64},
    offdiag_val::Float64,
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
)
    return (Y, X) -> begin
        for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], offdiag_val * sum_val)
        end
    end
end
