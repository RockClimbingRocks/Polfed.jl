

"""
    optimized_mapping!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy::Parallelization) -> Function

Build an optimized in-place mapping callback `f!(Y, X)` from packed diagonal and
off-diagonal data.
"""
function optimized_mapping!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy::Parallelization)
    opt_map! = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> mapping!(Y, X, diagonals, offdiagonals, parallel_strategy)

    return opt_map!
end


"""
    mapping!(Y, X, diagonals, offdiagonals, loop) -> nothing

Apply packed-value mapping in-place.

Overloads support vector and matrix state with either:
- a single off-diagonal value bucket tuple, or
- a vector of value buckets.

`Y` is mutated and `X` is treated as read-only.
"""
function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
    parallel_strategy::Parallelization,
)
    loop = make_loop(use_threads_in_loop(parallel_strategy))
    (val, offdiags_flatten, start_indices) = offdiagonals
    loop(eachindex(X), @inline i -> begin
        Y_i = mapping_state_i(X, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds Y[i] = Y_i
    end)

    return nothing
end



"""Vector method of `mapping!` for multiple off-diagonal value buckets."""
function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Number, Vector{Int}, Vector{Int}}},
    parallel_strategy::Parallelization,
)
    loop = make_loop(use_threads_in_loop(parallel_strategy))
    @. Y = diagonals * X
    
    for (val, offdiags_flatten, start_indices) in offdiagonals
        loop(eachindex(X), @inline i -> begin
            Y_off_val_i = mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)
            @inbounds Y[i] += Y_off_val_i
        end)
    end

    return nothing
end


@inline function mapping_columnwise!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
)
    @assert size(Y) == size(X)
    @assert length(diagonals) == size(X, 1)

    for col in axes(X, 2)
        mapping!(view(Y, :, col), view(X, :, col), diagonals, offdiagonals, parallel_strategy)
    end

    return nothing
end


"""Matrix/block method of `mapping!` applied column-wise."""
function mapping!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
)
    return mapping_columnwise!(Y, X, diagonals, offdiagonals, parallel_strategy)
end

"""Single-process two-level threading for block optimized mapping."""
function mapping!(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::MulColsParallel,
) where {T<:Number}
    parallel_strategy.nt_per_col <= 1 && return mapping_columnwise!(Y, X, diagonals, offdiagonals, parallel_strategy)

    @assert size(Y) == size(X)
    @assert length(diagonals) == size(X, 1)

    threaded_column_chunks!(size(X, 1), size(X, 2), parallel_strategy.nt_per_col) do start_row, end_row, col
        X_col = view(X, :, col)
        Y_col = view(Y, :, col)

        @inbounds for i in start_row:end_row
            Y_col[i] = mapping_state_i(X_col, i, diagonals, offdiagonals)
        end
    end

    return nothing
end
