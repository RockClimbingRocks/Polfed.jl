# Moments.jl

const permutation = SVector{3,Int64}(2,3,1)

# ==============================================================================
# 1. Main dispatcher function
# This is the function you will call from your `getdos!` script.
# It will automatically choose the correct implementation based on the type of `pu`.
# ==============================================================================
function dos_moments(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Real}, pu::ProcessingUnit)
    # This acts as a switch. If `pu` is a GPU, it calls the serial version.
    # If `pu` is a CPU, it calls the parallel version.
    return _dos_moments_impl(f!, N, R, hilbertspacedim, E, pu)
end


# ==============================================================================
# 2. GPU (and generic fallback) implementation - UNCHANGED
# This is your original code, just renamed. It will be used for the GPU.
# ==============================================================================
function _dos_moments_impl(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Real}, pu::GPU)
    # This is the original, serial implementation that works well on the GPU.
    vecs = [pu.Vector{E}(undef, hilbertspacedim) for _ in 1:3]
    μs = zeros(Float64, N)

    r = pu.Vector{E}(undef, hilbertspacedim)
    for _ in 1:R
        r .= pu.randn(hilbertspacedim)
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
function _dos_moments_impl(f!::Function, N::Int, R::Int, hilbertspacedim::Int, E::Type{<:Real}, pu::CPU)
    
    # Create a separate moment accumulator for each thread to prevent race conditions.
    μs_per_thread = [zeros(Float64, N) for _ in 1:Threads.nthreads()]

    # The main loop is now parallelized.
    # Each thread will execute a portion of the `1:R` iterations.
    Threads.@threads for _ in 1:R
        # --- Thread-Local Storage ---
        # Get the current thread's ID
        tid = Threads.threadid()
        # Get the dedicated moment vector for this thread
        μs_local = μs_per_thread[tid] 
        
        # Each thread needs its own temporary vectors to avoid conflicts.
        # This is crucial for correctness.
        vecs_local = [Vector{E}(undef, hilbertspacedim) for _ in 1:3]
        r_local = Vector{E}(undef, hilbertspacedim)
        # --- End Thread-Local Storage ---

        # The core calculation is the same as before, but on local variables.
        r_local .= randn(E, hilbertspacedim)
        r_local .*= 1/norm(r_local)
        
        # The `trace!` function is UNCHANGED, but now operates on thread-local data.
        trace!(f!, r_local, μs_local, vecs_local)
    end

    # --- Reduction Step ---
    # After the parallel loop, sum the results from all threads into one final vector.
    μs = sum(μs_per_thread)

    μs .*= hilbertspacedim/R
    return μs
end


# ==============================================================================
# The `trace!` function remains COMPLETELY UNCHANGED
# It is thread-safe because it only modifies the arrays passed into it.
# ==============================================================================
function trace!(f!::Function, α::AbstractVector{<:Number}, μs::AbstractVector{<:Number}, vecs::Vector{<:AbstractVecOrMat{<:Number}})
    # Dont change the order because it might depand on it! ( if α == β)

    vecs[1] .= α
    μs[1] += α⋅vecs[1]

    f!(vecs[2], vecs[1])
    μs[2] += α⋅vecs[2]

    for i in 2:length(μs)-1
        f!(vecs[3], vecs[2])
        @inbounds @. vecs[3] *= 2.
        @inbounds @. vecs[3] -= vecs[1]
        μs[i+1] += α⋅vecs[3];
        permute!(vecs, permutation);
    end
end