
"""
    optimized_clenshaw_recurrence_relation!(diagonals, offdiagonals, parallel_strategy) -> Function

Build an optimized Clenshaw recurrence callback compatible with POLFED's
internal recurrence signature.
"""
function optimized_clenshaw_recurrence_relation!(
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
)
    return (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, _k::Int, X::AbstractVecOrMat) -> begin
        map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals, parallel_strategy)
    end
end

"""
    map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals, parallel_strategy) -> nothing

Apply one in-place Clenshaw recurrence step using packed mapping data.

`b1` is overwritten with the new recurrence state.
"""
function map_with_clenshaw_recurrence_relation!(
    b1::AbstractVector{T},
    b2::AbstractVector{T},
    b3::AbstractVector{T},
    c::Real,
    X::AbstractVector{T},
    diagonals::AbstractVector,
    offdiagonals::Tuple{<:Number, Vector{Int}, Vector{Int}},
    parallel_strategy::Parallelization,
) where {T<:Number}
    loop = make_loop(use_threads_in_loop(parallel_strategy))
    val, offdiags_flatten, start_indices = offdiagonals

    loop(eachindex(X), @inline i -> begin
        yi = mapping_state_i(b2, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds b1[i] = c * X[i] + 2 * yi - b3[i]
    end)

    return nothing
end

"""Vector method of `map_with_clenshaw_recurrence_relation!` for multi-value buckets."""
function map_with_clenshaw_recurrence_relation!(
    b1::AbstractVector{T},
    b2::AbstractVector{T},
    b3::AbstractVector{T},
    c::Real,
    X::AbstractVector{T},
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Number, Vector{Int}, Vector{Int}}},
    parallel_strategy::Parallelization,
) where {T<:Number}
    loop = make_loop(use_threads_in_loop(parallel_strategy))

    mapping!(b1, b2, diagonals, offdiagonals, parallel_strategy)

    loop(eachindex(X), @inline i -> begin
        @inbounds b1[i] = c * X[i] + 2 * b1[i] - b3[i]
    end)

    return nothing
end

@inline function clenshaw_recurrence_columnwise!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    b3::AbstractMatrix{T},
    c::Real,
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
) where {T<:Number}
    for col in axes(X, 2)
        map_with_clenshaw_recurrence_relation!(
            view(b1, :, col),
            view(b2, :, col),
            view(b3, :, col),
            c,
            view(X, :, col),
            diagonals,
            offdiagonals,
            parallel_strategy,
        )
    end

    return nothing
end

"""Matrix/block method of `map_with_clenshaw_recurrence_relation!` applied column-wise."""
function map_with_clenshaw_recurrence_relation!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    b3::AbstractMatrix{T},
    c::Real,
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
) where {T<:Number}
    return clenshaw_recurrence_columnwise!(b1, b2, b3, c, X, diagonals, offdiagonals, parallel_strategy)
end

"""Single-process two-level threading for optimized block Clenshaw recurrence."""
function map_with_clenshaw_recurrence_relation!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    b3::AbstractMatrix{T},
    c::Real,
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::MulColsParallel,
) where {T<:Number}
    parallel_strategy.nt_per_col <= 1 && return clenshaw_recurrence_columnwise!(b1, b2, b3, c, X, diagonals, offdiagonals, parallel_strategy)

    threaded_column_chunks!(size(X, 1), size(X, 2), parallel_strategy.nt_per_col) do start_row, end_row, col
        b1_col = view(b1, :, col)
        b2_col = view(b2, :, col)
        b3_col = view(b3, :, col)
        X_col = view(X, :, col)

        @inbounds for i in start_row:end_row
            yi = mapping_state_i(b2_col, i, diagonals, offdiagonals)
            b1_col[i] = c * X_col[i] + 2 * yi - b3_col[i]
        end
    end

    return nothing
end
