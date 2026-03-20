

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
    use_threads_in_loop = UseThreadsInLoop[typeof(parallel_strategy)]
    loop = make_loop(use_threads_in_loop)

    return (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Real, _k::Int, X::AbstractVecOrMat) -> begin
        map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals, loop)
    end
end

"""
    map_with_clenshaw_recurrence_relation!(b1, b2, b3, c, X, diagonals, offdiagonals, loop) -> nothing

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
    loop::Function,
) where {T<:Number}

    (val, offdiags_flatten, start_indices) = offdiagonals

    # println("Using optimized mapping clenshaw mapping TUPLED!!")
    loop(eachindex(X), @inline i -> begin
        yi = mapping_state_i(b2, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds b1[i] = c*X[i] + 2*yi - b3[i]
    end)
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
    loop::Function,
) where {T<:Number}

    mapping!(b1,b2,diagonals,offdiagonals,loop)

    loop(eachindex(X), @inline i -> begin
        @inbounds b1[i] = c*X[i] + 2*b1[i] - b3[i]
    end)

    # @. b1 = c*X + 2*b1 - b3

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
    loop::Function,
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
            loop,
        )
    end

end
