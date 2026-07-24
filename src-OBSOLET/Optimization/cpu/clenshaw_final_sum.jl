
"""
    optimized_clenshaw_final_sum!(diagonals, offdiagonals, parallel_strategy) -> Function

Build an optimized callback for the terminal Clenshaw combination step.
"""
function optimized_clenshaw_final_sum!(
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
)
    return (b1::AbstractVecOrMat, b2::AbstractVecOrMat, c::Real, Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
        map_with_clenshaw_final_sum!(b1, b2, c, Y, X, diagonals, offdiagonals, parallel_strategy)
    end
end


"""
    map_with_clenshaw_final_sum!(b1, b2, c, Y, X, diagonals, offdiagonals, parallel_strategy) -> nothing

Apply the final in-place Clenshaw accumulation step using packed mapping data.

`Y` is mutated to contain the final mapped combination.
"""
function map_with_clenshaw_final_sum!(
    b1::AbstractVector{T},
    b2::AbstractVector{T},
    c::Real,
    Y::AbstractVector{T},
    X::AbstractVector{T},
    diagonals::AbstractVector,
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
    parallel_strategy::Parallelization,
) where {T<:Number}
    loop = make_loop(use_threads_in_loop(parallel_strategy))
    val, offdiags_flatten, start_indices = offdiagonals

    loop(eachindex(X), @inline i -> begin
        yi = mapping_state_i(b1, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds Y[i] = c * X[i] + yi - b2[i]
    end)

    return nothing
end


"""Vector method of `map_with_clenshaw_final_sum!` for multi-value off-diagonal buckets."""
function map_with_clenshaw_final_sum!(
    b1::AbstractVector{T},
    b2::AbstractVector{T},
    c::Real,
    Y::AbstractVector{T},
    X::AbstractVector{T},
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Number, Vector{Int}, Vector{Int}}},
    parallel_strategy::Parallelization,
) where {T<:Number}
    loop = make_loop(use_threads_in_loop(parallel_strategy))

    mapping!(Y, b1, diagonals, offdiagonals, parallel_strategy)

    loop(eachindex(X), @inline i -> begin
        @inbounds Y[i] = c * X[i] + Y[i] - b2[i]
    end)

    return nothing
end


@inline function clenshaw_final_sum_columnwise!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    c::Real,
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
) where {T<:Number}
    for col in axes(X, 2)
        map_with_clenshaw_final_sum!(
            view(b1, :, col),
            view(b2, :, col),
            c,
            view(Y, :, col),
            view(X, :, col),
            diagonals,
            offdiagonals,
            parallel_strategy,
        )
    end

    return nothing
end


"""Matrix/block method of `map_with_clenshaw_final_sum!` applied column-wise."""
function map_with_clenshaw_final_sum!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    c::Real,
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
) where {T<:Number}
    return clenshaw_final_sum_columnwise!(b1, b2, Y, c, X, diagonals, offdiagonals, parallel_strategy)
end

"""Single-process two-level threading for optimized block Clenshaw final sum."""
function map_with_clenshaw_final_sum!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    c::Real,
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::MulColsParallel,
) where {T<:Number}
    parallel_strategy.nt_per_col <= 1 && return clenshaw_final_sum_columnwise!(b1, b2, Y, c, X, diagonals, offdiagonals, parallel_strategy)

    threaded_column_chunks!(size(X, 1), size(X, 2), parallel_strategy.nt_per_col) do start_row, end_row, col
        b1_col = view(b1, :, col)
        b2_col = view(b2, :, col)
        Y_col = view(Y, :, col)
        X_col = view(X, :, col)

        @inbounds for i in start_row:end_row
            yi = mapping_state_i(b1_col, i, diagonals, offdiagonals)
            Y_col[i] = c * X_col[i] + yi - b2_col[i]
        end
    end

    return nothing
end
