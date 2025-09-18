

const Offdiagonal   = Tuple{<:Number, Vector{Int}, Vector{Int}}
const Offdiagonals = Union{Offdiagonal, Vector{Offdiagonal}}

include("mapping.jl")
include("clenshaw_reocurrence_relation.jl")
include("clenshaw_final_sum.jl")




function optimize_spectral_transform(mat::AbstractMatrix{T}, config::SpectralTransformConfig) where {T<:Real}
    # Implement optimization logic here
    

    Emin = -1. 
    Emax = 1.
    spread = (Emax-Emin)/2
    center = (Emax+Emin)/2


    
    diagonals, offdiagonals = get_diags_and_offdiagonals_by_value(mat)

    diagonals_rescaled = (diagonals .- center)./spread
    offdiagonals_rescaled = [(val/spread, flat, starts) for (val, flat, starts) in offdiagonals]


    f!_opt = optimized_mapping!(diagonals, offdiagonals)
    f!_rescaled_opt = optimized_mapping!(diagonals_rescaled, offdiagonals_rescaled)

    crr_rescale = optimized_clenshaw_recurrence_relation!(diagonals_rescaled, offdiagonals_rescaled, config.parallelization)    

end



function get_diags_and_offdiagonals_by_value(mat::AbstractMatrix{T}; tol=1e-14, round_digits=15) where {T<:Real}
    dim = size(mat, 1)

    # Map: val => list of connections for each row
    value_to_conn = Dict{Float64, Vector{Vector{Int}}}()

    # Collect diagonal entries
    diagonals = Vector{Float64}(undef, dim)

    for i in 1:dim
        diagonals[i] = round(mat[i, i]; digits=round_digits)  # store diagonal value

        row_conns = Dict{Float64, Vector{Int}}()
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            if i == j  # skip diagonal
                continue
            end
            v = mat[i, j]
            if abs(v) < tol
                continue
            end
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

    # Now flatten each list
    offdiags = Tuple{Float64, Vector{Int}, Vector{Int}}[]
    for (v, conn_lists) in sort(collect(value_to_conn), by=first)
        flat = Int[]
        starts = Int[]
        idx = 1
        for js in conn_lists
            push!(starts, idx)
            append!(flat, js)
            idx += length(js)
        end
        push!(offdiags, (v, flat, starts))
    end

    return diagonals, offdiags
end


