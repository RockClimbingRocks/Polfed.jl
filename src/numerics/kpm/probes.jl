struct ProbeInfo
    dim::Int
    block_size::Int
    layout::Symbol
end

"""
    probe_info(template) -> ProbeInfo

Return the Hilbert-space dimension, number of probes per batch, and probe layout
represented by a KPM probe template.

- `Vector{E}(undef, dim)` stores one probe.
- `Matrix{E}(undef, dim, block)` stores column probes for `f!(Y, X) = H*X`.
- `Matrix{E}(undef, block, dim)` stores row probes for `f!(Y, X) = X*H`.

For general matrices, the larger dimension is treated as the Hilbert-space
dimension. Square matrices are interpreted as column-probe blocks.
"""
function probe_info(template::AbstractVector)
    dim = length(template)
    dim > 0 || throw(ArgumentError("KPM probe dimension must be positive."))
    return ProbeInfo(dim, 1, :vector)
end

function probe_info(template::AbstractMatrix)
    nrow, ncol = size(template)
    nrow > 0 && ncol > 0 || throw(ArgumentError("KPM probe dimensions must be positive, got $(size(template))."))

    if nrow == 1 && ncol > 1
        return ProbeInfo(ncol, nrow, :rows)
    end

    if nrow >= ncol
        return ProbeInfo(nrow, ncol, :columns)
    else
        return ProbeInfo(ncol, nrow, :rows)
    end
end

"""
    probe_dimension(template) -> Int

Return the Hilbert-space dimension represented by a KPM probe template.
"""
function probe_dimension(template::AbstractVector)
    return probe_info(template).dim
end

function probe_dimension(template::AbstractMatrix)
    return probe_info(template).dim
end

_random_normal!(x, pu::ProcessingUnit) = _random_normal!(x, pu, Random.default_rng())

_random_normal!(x, ::CPU, rng::AbstractRNG) = (randn!(rng, x); x)

"""
    _random_normal!(x, pu::GPU, rng) -> x

GPU probes come from the DEVICE generator; `rng` is accepted for a uniform signature and
ignored, because `pu.randn` allocates on the device and takes no host generator. A GPU DoS is
therefore not reproducible from a host seed — say so rather than let a caller who seeded
carefully assume otherwise.
"""
_random_normal!(x, pu::GPU, ::AbstractRNG) = (x .= pu.randn(eltype(x), size(x)...); x)

"""
    _probe_slot(x, info, p) -> AbstractArray

Probe `p` of the block `x`: the vector itself, column `p`, or row `p` (per `info.layout`). A
one-probe block is returned as `vec(x)` (same memory), so a single-probe GPU block never goes
through a strided view.
"""
_probe_slot(x::AbstractVector, ::ProbeInfo, ::Int) = x
function _probe_slot(x::AbstractMatrix, info::ProbeInfo, p::Int)
    info.block_size == 1 && return vec(x)
    info.layout === :columns && return view(x, :, p)
    info.layout === :rows    && return view(x, p, :)
    throw(ArgumentError("Unknown KPM probe layout: $(info.layout)."))
end

"""
    _draw_probe_block!(block, scratch, info, active, pu, rng) -> block

Fill the first `active` probe slots of `block` with unit-norm Gaussian probes and zero the rest.

Each probe is drawn on its own — `randn!` into the contiguous length-`dim` `scratch`, normalized,
then copied into its slot — never one `randn!` over the whole block. A block fill is a different
random stream (MersenneTwister fills an `Array{Float64}` in bulk, and a strided slot is not an
`Array` at all), so the probes would change with the block width. Drawn this way, probe `r` is
the `r`-th vector drawn from `rng` whatever the width.
"""
function _draw_probe_block!(block::AbstractVecOrMat, scratch::AbstractVector, info::ProbeInfo,
                            active::Int, pu::ProcessingUnit, rng::AbstractRNG)
    1 <= active <= info.block_size ||
        throw(ArgumentError("Active KPM probes must be in 1:$(info.block_size), got $active."))

    for p in 1:active
        _random_normal!(scratch, pu, rng)
        nrm = norm(scratch)
        nrm > 0 || throw(ArgumentError("Generated a zero KPM random probe; cannot normalize it."))
        scratch ./= nrm
        _probe_slot(block, info, p) .= scratch
    end
    _zero_inactive_probes!(block, info, active)
    return block
end

_zero_inactive_probes!(x::AbstractVector, ::ProbeInfo, ::Int) = x

function _zero_inactive_probes!(x::AbstractMatrix, info::ProbeInfo, active::Int)
    active == info.block_size && return x

    if info.layout === :columns
        @views fill!(x[:, active+1:info.block_size], zero(eltype(x)))
    elseif info.layout === :rows
        @views fill!(x[active+1:info.block_size, :], zero(eltype(x)))
    else
        throw(ArgumentError("Unknown KPM probe layout: $(info.layout)."))
    end

    return x
end
