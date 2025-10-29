using SparseArrays
using LinearAlgebra

function construct_XXZ_matrix(L::Int, Δ::Real, Nup::Int)
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
            push!(rows, col); push!(cols, col); push!(vals, Δ * SzSz)
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



function get_diags_and_offdiagonals_single_value(mat::AbstractMatrix{T}; tol=1e-13, round_digits=14) where {T<:Real}
    dim = size(mat, 1)
    diagonals = [round(mat[i, i]; digits=round_digits) for i in 1:dim]
    flat = Int[]
    starts = Int[]
    idx = 1
    offdiag_val::Union{Nothing, T} = nothing

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
    return (Y,X) -> begin
        for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop  = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1]-1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], offdiag_val * sum_val)
        end
    end
end