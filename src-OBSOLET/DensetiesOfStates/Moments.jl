# Moments.jl

const permutation = SVector{3,Int64}(2,3,1)

"""
    _thread_slots() -> Int

Return the number of valid storage slots needed for indexing by `Threads.threadid()`.

On Julia versions with separate interactive and default thread pools, `threadid()`
is global across pools, so `Threads.nthreads()` alone is not a safe upper bound.
"""
function _thread_slots()
    try
        return Threads.nthreads(:default) + Threads.nthreads(:interactive)
    catch
        return Threads.nthreads()
    end
end

# ==============================================================================
# 1. Main dispatcher function
# This is the function you will call from your `getdos!` script.
# It will automatically choose the correct implementation based on the type of `pu`.
# ==============================================================================
"""
    dos_moments(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Number}, pu::ProcessingUnit) -> AbstractVector

Compute KPM moments for DoS estimation.

# Arguments
- `f!`: In-place mapping callback `f!(Y, X)`.
- `N`: Number of moments.
- `R`: Number of random probe vectors.
- `hilbertspacedim`: Vector length.
- `E`: Element type used for probe vectors.
- `pu`: Processing unit (`CPU` or `GPU`).

# Returns
- Moment vector `μ` of length `N`.
"""
function dos_moments(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Number}, pu::ProcessingUnit)
    # This acts as a switch. If `pu` is a GPU, it calls the serial version.
    # If `pu` is a CPU, it calls the parallel version.
    return _dos_moments_impl(f!, N, R, hilbertspacedim, E, pu)
end


# ==============================================================================
# 2. GPU (and generic fallback) implementation - UNCHANGED
# This is your original code, just renamed. It will be used for the GPU.
# ==============================================================================
"""
    _dos_moments_impl(..., pu::GPU) -> AbstractVector

GPU-oriented implementation of DoS moment accumulation.
"""
function _dos_moments_impl(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Number}, pu::GPU)
    # This is the original, serial implementation that works well on the GPU.
    vecs = [pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3]
    S = real(E)
    μs = zeros(S, N)

    r = pu.Vector{E}(undef, hilbertspacedim)
    for _ in 1:R
        r .= pu.randn(E, hilbertspacedim)
        r .*= 1/norm(r)
        trace!(f!, r, μs, vecs) # Adds results directly to μs
    end

    μs .*= hilbertspacedim/R
    return μs
end


# ==============================================================================
# 3. CPU-specific PARALLEL implementation
# This version uses multi-threading for high performance on CPUs.
# ==============================================================================
"""
    _dos_moments_impl(..., pu::CPU) -> AbstractVector

CPU-oriented threaded implementation of DoS moment accumulation.

Uses per-thread local accumulators and reduces them at the end.
"""
function _dos_moments_impl(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Number}, pu::CPU)
    S = real(E)
    nslots = _thread_slots()
    μs_per_thread = [zeros(S, N) for _ in 1:nslots]
    vecs_per_thread = [[Vector{E}(undef, hilbertspacedim) for _ in 1:3] for _ in 1:nslots]
    r_per_thread = [Vector{E}(undef, hilbertspacedim) for _ in 1:nslots]

    # Use static scheduling so each worker reuses its dedicated scratch buffers.
    Threads.@threads :static for _ in 1:R
        tid = Threads.threadid()
        μs_local = μs_per_thread[tid]
        vecs_local = vecs_per_thread[tid]
        r_local = r_per_thread[tid]

        r_local .= randn(E, hilbertspacedim)
        r_local .*= inv(norm(r_local))
        trace!(f!, r_local, μs_local, vecs_local)
    end

    μs = reduce(+, μs_per_thread; init=zeros(S, N))
    μs .*= hilbertspacedim/R
    return μs
end


# ==============================================================================
# The `trace!` function remains COMPLETELY UNCHANGED
# It is thread-safe because it only modifies the arrays passed into it.
# ==============================================================================
"""
    trace!(f!::Function, α::AbstractVector, μs::AbstractVector, vecs::Vector) -> nothing

Accumulate Chebyshev moments in-place for a single normalized random vector.

`μs` is mutated by adding contributions from `α`; `vecs` is used as internal
three-buffer recurrence workspace.
"""
function trace!(f!::Function, α::AbstractVector{<:Number}, μs::AbstractVector{<:Real}, vecs::Vector{<:AbstractVecOrMat{<:Number}})
    # Dont change the order because it might depand on it! ( if α == β)

    vecs[1] .= α
    μs[1] += real(α ⋅ vecs[1])

    f!(vecs[2], vecs[1])
    μs[2] += real(α ⋅ vecs[2])

    for i in 2:length(μs)-1
        f!(vecs[3], vecs[2])
        @inbounds @. vecs[3] *= 2 * one(eltype(vecs[3]))
        @inbounds @. vecs[3] -= vecs[1]
        μs[i+1] += real(α ⋅ vecs[3]);
        permute!(vecs, permutation);
    end
end
