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

function probe_block_size(template::AbstractVecOrMat)
    return probe_info(template).block_size
end

function _random_normal!(x, ::CPU)
    randn!(x)
    return x
end

function _random_normal!(x, pu::GPU)
    x .= pu.randn(eltype(x), size(x)...)
    return x
end

function _random_unit_probe!(x, pu::ProcessingUnit, info::ProbeInfo = probe_info(x), active::Int = info.block_size)
    1 <= active <= info.block_size ||
        throw(ArgumentError("Active KPM probes must be in 1:$(info.block_size), got $active."))

    _random_normal!(x, pu)
    _zero_inactive_probes!(x, info, active)
    _normalize_active_probes!(x, info, active)
    return x
end

function _normalize_active_probes!(x::AbstractVector, ::ProbeInfo, active::Int)
    active == 1 || throw(ArgumentError("Vector KPM probes can only have one active probe, got $active."))
    nrm = norm(x)
    nrm > 0 || throw(ArgumentError("Generated a zero KPM random probe; cannot normalize it."))
    x ./= nrm
    return x
end

function _normalize_active_probes!(x::AbstractMatrix, info::ProbeInfo, active::Int)
    if info.layout === :columns
        @views for j in 1:active
            probe = x[:, j]
            nrm = norm(probe)
            nrm > 0 || throw(ArgumentError("Generated a zero KPM random probe; cannot normalize it."))
            probe ./= nrm
        end
    elseif info.layout === :rows
        @views for i in 1:active
            probe = x[i, :]
            nrm = norm(probe)
            nrm > 0 || throw(ArgumentError("Generated a zero KPM random probe; cannot normalize it."))
            probe ./= nrm
        end
    else
        throw(ArgumentError("Unknown KPM probe layout: $(info.layout)."))
    end
    return x
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
