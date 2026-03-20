
"""
    optimized_clenshaw_final_sum!(diagonals, offdiagonals, parallel_strategy) -> Function

Build an optimized callback for the terminal Clenshaw combination step.
"""
function optimized_clenshaw_final_sum!(
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    parallel_strategy::Parallelization,
)
    use_threads_in_loop = UseThreadsInLoop[typeof(parallel_strategy)]
    loop = make_loop(use_threads_in_loop)

    return (b1::AbstractVecOrMat, b2::AbstractVecOrMat, c::Real, Y::AbstractVecOrMat, X::AbstractVecOrMat) -> begin
        map_with_clenshaw_final_sum!(b1, b2, c, Y, X, diagonals, offdiagonals, loop)
    end
end


"""
    map_with_clenshaw_final_sum!(b1, b2, c, Y, X, diagonals, offdiagonals, loop) -> nothing

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
    loop::Function,
) where {T<:Number}

    (val, offdiags_flatten, start_indices) = offdiagonals

    loop(eachindex(X), @inline i -> begin
        yi = mapping_state_i(b1, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds Y[i] = c*X[i] + yi - b2[i]
    end)

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
    loop::Function,
) where {T<:Number} 

    mapping!(Y, b1, diagonals, offdiagonals, loop)

    loop(eachindex(X), @inline i -> begin
        @inbounds Y[i] = c*X[i] + Y[i] - b2[i]
    end)
end






"""Matrix/block method of `map_with_clenshaw_final_sum!` applied column-wise."""
function map_with_clenshaw_final_sum!(
    b1::AbstractMatrix{T},
    b2::AbstractMatrix{T},
    b3::AbstractMatrix{T},
    c::Real,
    X::AbstractMatrix{T},
    diagonals::AbstractVector,
    offdiagonals::Offdiagonals,
    loop::Function,
) where {T<:Number}

    for col in axes(X, 2)
        map_with_clenshaw_final_sum!(
            view(b1, :, col),
            view(b2, :, col),
            view(b3, :, col),
            c,
            view(X, :, col),
            diagonals,
            offdiagonals,
            loop,
        )
    end
end
