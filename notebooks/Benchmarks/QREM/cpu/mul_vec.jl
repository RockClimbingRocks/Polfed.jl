function get_offdiagonals_by_value(qrem::QREM; tol=1e-14, round_digits=15)
    H = construct_matrix(qrem)  # Should already be sparse
    N = size(H, 1)

    # Map: val => list of connections for each row
    value_to_conn = Dict{Float64, Vector{Vector{Int}}}()

    for i in 1:N
        row_conns = Dict{Float64, Vector{Int}}()
        for col in nzrange(H, i)
            j = rowvals(H)[col]
            if i == j  # skip diagonal
                continue
            end
            v = H[i, j]
            if abs(v) < tol
                continue
            end
            v_rounded = round(v; digits=round_digits)
            push!(get!(row_conns, v_rounded, Int[]), j)
        end
        for (v, js) in row_conns
            if !haskey(value_to_conn, v)
                value_to_conn[v] = [Int[] for _ in 1:N]
            end
            value_to_conn[v][i] = js
        end
    end

    # Now flatten each list
    result = Tuple{Float64, Vector{Int}, Vector{Int}}[]
    for (v, conn_lists) in sort(collect(value_to_conn), by=first)
        flat = Int[]
        starts = Int[]
        idx = 1
        for js in conn_lists
            push!(starts, idx)
            append!(flat, js)
            idx += length(js)
        end
        push!(result, (v, flat, starts))
    end

    return result
end


function map_vec_cpu(qrem::QREM; a::Float64=1.0, b::Float64=0.0)

    qrem_rescaled = QREM(
        "qrem",
        qrem.L,
        qrem.hx / a,
        qrem.spin,
        (qrem.diags .- b) ./ a
    )

    offdiags = get_offdiagonals_by_value(qrem_rescaled)  # use your generalized function

    return @inline (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
        mapvec_with_qrem!(Y, X, qrem_rescaled.diags, offdiags)
    end
end


function mapvec_with_qrem!(
    Y::AbstractVector,
    X::AbstractVector,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
)

    @. Y = diags * X
    
    for (val, flat, start_inds) in offdiagonals
        # total_sum_val = 0.
        for i in eachindex(diags)
            @inbounds start = start_inds[i]
            @inbounds stop = i == length(start_inds) ? length(flat) : start_inds[i + 1] - 1

            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[flat[j]]
            end

            # total_sum_val += sum_val * val

            @inbounds Y[i] += sum_val * val
        end
    end
end




function mapvec_with_qrem!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diags::AbstractVector{Float64},
    offdiagonals::AbstractVector,
)
    @assert size(X, 1) == length(diags)
    @assert size(Y) == size(X)

    n, m = size(X)
    for col in 1:m
        mapvec_with_qrem!(view(Y, :, col), view(X, :, col), diags, offdiagonals)
    end
end






# function mapvec_with_qrem_2(
#     Y::AbstractVector,
#     X::AbstractVector,
#     diags::AbstractVector{Float64},
#     offdiagonals::AbstractVector,
# )

#     @simd for i in eachindex(diags)
#         total_sum_val = 0.
#         for (val, flat, start_inds) in offdiagonals
#             @inbounds start = start_inds[i]
#             @inbounds stop = i == length(start_inds) ? length(flat) : start_inds[i + 1] - 1

#             sum_val = 0.0
#             for j in start:stop
#                 @inbounds sum_val += X[flat[j]]
#             end

#             total_sum_val += sum_val * val

#         end
#         @inbounds Y[i] = diags[i] * X[i] + total_sum_val
#     end
# end

