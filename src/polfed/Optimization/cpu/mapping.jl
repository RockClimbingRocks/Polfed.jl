

function optimized_mapping!(diagonals::AbstractVector, offdiagonals::Offdiagonals, parallel_strategy::Parallelization)
    use_threads_in_loop = UseThreadsInLoop[typeof(parallel_strategy)]
    loop = make_loop(use_threads_in_loop)

    opt_map! = (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> mapping!(Y, X, diagonals, offdiagonals, loop)

    return opt_map!
end


function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Tuple{<:Real, Vector{Int}, Vector{Int}},
    loop::Function,   # the loop is injected
)
    (val, offdiags_flatten, start_indices) = offdiagonals
    loop(eachindex(X), @inline i -> begin
        Y_i = mapping_state_i(X, i, diagonals, val, offdiags_flatten, start_indices)
        @inbounds Y[i] = Y_i
    end)
end



function mapping!(
    Y::AbstractVector,
    X::AbstractVector,
    diagonals::AbstractVector,
    offdiagonals::Vector{<:Tuple{<:Real, Vector{Int}, Vector{Int}}},
    loop::Function,   
)
    @. Y = diagonals * X
    
    for (val, offdiags_flatten, start_indices) in offdiagonals
        loop(eachindex(X), @inline i -> begin
            Y_off_val_i = mapping_offdiagonals_state_i(X, i, val, offdiags_flatten, start_indices)
            @inbounds Y[i] += Y_off_val_i
        end)
    end
end



function mapping!(
    Y::AbstractMatrix,
    X::AbstractMatrix,
    diagonals::AbstractVector,
    offdiagonals,
    loop::Function,
)
    @assert size(Y) == size(X)
    @assert length(diagonals) == size(X, 1)

    for col in axes(X, 2)
        mapping!(view(Y, :, col), view(X, :, col), diagonals, offdiagonals, loop)
    end
end

