# Chebyshev moment estimation for KPM.
#
# The only shape rule is: `template` has the same shape that `f!(out, in)` expects.
# A vector template supports `H*x`; a `dim x block` matrix supports `H*X`; a
# `block x dim` matrix supports `X*H`. The recurrence itself is identical for all
# three cases.

"""
    ChebyshevWorkspace(template)

Three buffers shaped like `template`, reused by [`trace_doubling!`](@ref).
"""
struct ChebyshevWorkspace{A}
    prev::A
    curr::A
    next::A
end

ChebyshevWorkspace(template) =
    ChebyshevWorkspace(similar(template), similar(template), similar(template))

"""
    trace_doubling!(f!, probe, moments, work) -> moments

Accumulate Chebyshev moments for one normalized probe into `moments`.

`f!(out, in)` must write the mapped vector/matrix into `out` and preserve the
shape of `in`. Correctness assumes the mapped operator is Hermitian and already
scaled to the Chebyshev interval.
"""
function trace_doubling!(
    f!,
    probe::AbstractVecOrMat{<:Number},
    moments::AbstractVector{<:Real},
    work::ChebyshevWorkspace,
)
    N = length(moments)
    N == 0 && return moments

    prev = work.prev
    curr = work.curr
    next = work.next

    copyto!(prev, probe)
    m0 = real(dot(probe, prev))
    moments[1] += m0
    N == 1 && return moments

    f!(curr, prev)
    m1 = real(dot(probe, curr))
    moments[2] += m1

    max_step = cld(N - 1, 2)
    @inbounds for n in 1:max_step
        even_k = 2 * n
        odd_k = 2 * n - 1

        if even_k <= N - 1
            moments[even_k + 1] += 2 * real(dot(curr, curr)) - m0
        end
        if n >= 2 && odd_k <= N - 1
            moments[odd_k + 1] += 2 * real(dot(curr, prev)) - m1
        end

        if n < max_step
            f!(next, curr)
            @. next = 2 * next - prev
            prev, curr, next = curr, next, prev
        end
    end

    return moments
end

function _check_moment_args(N::Int, R::Int)
    N >= 1 || throw(ArgumentError("Number of moments N must be >= 1, got $N."))
    R >= 1 || throw(ArgumentError("Number of probes R must be >= 1, got $R."))
    return nothing
end

function _moments_one_worker(f!, N::Int, R::Int, template, pu::ProcessingUnit, info::ProbeInfo,
                             rng::AbstractRNG)
    S = real(eltype(template))
    local_moments = zeros(S, N)
    probe = similar(template)
    work = ChebyshevWorkspace(template)

    for first_probe in 1:info.block_size:R
        active = min(info.block_size, R - first_probe + 1)
        _random_unit_probe!(probe, pu, info, active, rng)
        trace_doubling!(f!, probe, local_moments, work)
    end

    return local_moments
end

"""
    kpm_moments(f!, N, R, template, pu) -> Vector

Estimate the first `N` Chebyshev moments using `R` random probes.

`template` is only a shape and element-type example:

- `Vector{E}(undef, dim)` for `f!(y, x) = H*x`
- `Matrix{E}(undef, dim, block)` for `f!(Y, X) = H*X`
- `Matrix{E}(undef, block, dim)` for `f!(Y, X) = X*H`

For matrix templates, each column or row is one probe. `R` is the total number
of probes, so a block size of `2` with `R == 100` performs `50` matrix
applications per recurrence step.

`rng` seeds the probes. The estimate is stochastic, so two calls with the default generator
give different moments — pass a seeded generator when the result has to be reproducible. On a
[`GPU`](@ref) the device generator is used and `rng` is ignored (see [`_random_normal!`](@ref)).
"""
function kpm_moments(
    f!,
    N::Int,
    R::Int,
    template::AbstractVecOrMat{E},
    pu::ProcessingUnit;
    rng::AbstractRNG = Random.default_rng(),
) where {E<:Number}
    _check_moment_args(N, R)
    info = probe_info(template)

    moments = _moments_one_worker(f!, N, R, template, pu, info, rng)
    moments .*= info.dim / R
    return moments
end
