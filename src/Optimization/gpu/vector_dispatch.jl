const GpuIndexVector = CuVector{<:Integer}
const GpuOffdiagonal = Tuple{<:Number, <:GpuIndexVector, <:GpuIndexVector}
const GpuOffdiagonals = Union{GpuOffdiagonal, Vector{<:GpuOffdiagonal}}

@inline function gpu_launch_config(n::Integer)
    threads = min(256, max(1, Int(n)))
    blocks = cld(Int(n), threads)
    return threads, blocks
end

function mapping_single_bucket_kernel!(
    Y,
    X,
    D,
    val,
    flat,
    starts,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = length(Y)

    if i <= n
        start_idx = starts[i]
        stop_idx = (i == length(starts)) ? length(flat) : starts[i + 1] - 1

        acc = zero(eltype(Y))
        @inbounds for ptr in start_idx:stop_idx
            acc += X[flat[ptr]]
        end

        @inbounds Y[i] = muladd(D[i], X[i], val * acc)
    end

    return nothing
end

function mapping_add_bucket_kernel!(
    Y,
    X,
    val,
    flat,
    starts,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = length(Y)

    if i <= n
        start_idx = starts[i]
        stop_idx = (i == length(starts)) ? length(flat) : starts[i + 1] - 1

        acc = zero(eltype(Y))
        @inbounds for ptr in start_idx:stop_idx
            acc += X[flat[ptr]]
        end

        @inbounds Y[i] += val * acc
    end

    return nothing
end

function gpu_mapping!(
    Y::AbstractVector{T},
    X::AbstractVector{T},
    diagonals::CuVector{T},
    offdiagonals::GpuOffdiagonal,
) where {T<:Real}
    val, flat, starts = offdiagonals
    threads, blocks = gpu_launch_config(length(Y))
    CUDA.@cuda threads=threads blocks=blocks mapping_single_bucket_kernel!(Y, X, diagonals, T(val), flat, starts)
    return nothing
end

function gpu_mapping!(
    Y::AbstractVector{T},
    X::AbstractVector{T},
    diagonals::CuVector{T},
    offdiagonals::Vector{<:GpuOffdiagonal},
) where {T<:Real}
    @. Y = diagonals * X

    threads, blocks = gpu_launch_config(length(Y))
    for (val, flat, starts) in offdiagonals
        CUDA.@cuda threads=threads blocks=blocks mapping_add_bucket_kernel!(Y, X, T(val), flat, starts)
    end

    return nothing
end

function gpu_mapping!(
    Y::AbstractMatrix{T},
    X::AbstractMatrix{T},
    diagonals::CuVector{T},
    offdiagonals::GpuOffdiagonals,
) where {T<:Real}
    @assert size(Y) == size(X)

    for col in axes(X, 2)
        gpu_mapping!(view(Y, :, col), view(X, :, col), diagonals, offdiagonals)
    end

    return nothing
end

function optimized_mapping!(
    diagonals::CuVector{T},
    offdiagonals::GpuOffdiagonals,
    ::Parallelization,
) where {T<:Real}
    return (Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}) -> gpu_mapping!(Y, X, diagonals, offdiagonals)
end

function optimized_clenshaw_recurrence_relation!(
    diagonals::CuVector{T},
    offdiagonals::GpuOffdiagonals,
    parallel_strategy::Parallelization,
) where {T<:Real}
    map_opt! = optimized_mapping!(diagonals, offdiagonals, parallel_strategy)

    return (b1::AbstractVecOrMat{T}, b2::AbstractVecOrMat{T}, b3::AbstractVecOrMat{T}, c::Real, _k::Int, X::AbstractVecOrMat{T}) -> begin
        map_opt!(b1, b2)
        @. b1 = c * X + 2 * b1 - b3
        nothing
    end
end

function optimized_clenshaw_final_sum!(
    diagonals::CuVector{T},
    offdiagonals::GpuOffdiagonals,
    parallel_strategy::Parallelization,
) where {T<:Real}
    map_opt! = optimized_mapping!(diagonals, offdiagonals, parallel_strategy)

    return (b1::AbstractVecOrMat{T}, b2::AbstractVecOrMat{T}, c::Real, Y::AbstractVecOrMat{T}, X::AbstractVecOrMat{T}) -> begin
        map_opt!(Y, b1)
        @. Y = c * X + Y - b2
        nothing
    end
end
