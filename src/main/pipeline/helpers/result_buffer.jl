const _RESULT_ATOL = 1e-9

"""
    _DenseResultBuffer

Preallocated result storage for pipelines whose maximum result count is known. Eigenvectors
are written directly into one layout-aware matrix. With `target === nothing`, pairs append
until `capacity`; otherwise the buffer retains the closest pairs to `target` in sorted order.
"""
mutable struct _DenseResultBuffer{M<:AbstractMatrix,L<:Layout}
    vals::Vector{Float64}
    vecs::M
    used::Int
    layout::L
    target::Union{Nothing,Float64}
    duplicate_atol::Float64
end

"""
    _WindowResultBuffer

Dynamically sized storage for window pipelines, which do not know their result count in
advance. Each accepted eigenvector is copied into an independent vector, releasing the
transient Lanczos result matrix before the next patch.
"""
mutable struct _WindowResultBuffer{V<:AbstractVector,L<:Layout}
    vals::Vector{Float64}
    vecs::Vector{V}
    layout::L
    duplicate_atol::Float64
end

"""
    _dense_result_buffer(mp, capacity, layout; target, duplicate_atol)

Allocate one backend-native eigenvector matrix with `capacity` row- or column-oriented slots.
"""
function _dense_result_buffer(
    mp::MappingPlan,
    capacity::Integer,
    layout::Layout;
    target = nothing,
    duplicate_atol::Real,
)
    n = Int(capacity)
    n > 0 || throw(ArgumentError("result capacity must be positive; got $capacity."))
    T, dim = eltype(mp.sys), mp.sys.dim
    vecs = layout isa RowLayout ?
        mp.sys.pu.Matrix{T}(undef, n, dim) :
        mp.sys.pu.Matrix{T}(undef, dim, n)
    return _DenseResultBuffer(
        Vector{Float64}(undef, n), vecs, 0, layout,
        isnothing(target) ? nothing : Float64(target),
        Float64(duplicate_atol),
    )
end

"""
    _window_result_buffer(mp, layout; duplicate_atol)

Create dynamic result storage whose individual vectors use the mapping plan's backend and type.
"""
function _window_result_buffer(mp::MappingPlan, layout::Layout; duplicate_atol::Real)
    V = mp.sys.pu.Vector{eltype(mp.sys)}
    return _WindowResultBuffer(Float64[], Vector{V}(), layout, Float64(duplicate_atol))
end

"""Return eigenvector `i` as a view in the orientation selected by `layout`."""
_result_vector_view(::ColumnLayout, vecs::AbstractMatrix, i::Int) = view(vecs, :, i)
_result_vector_view(::RowLayout, vecs::AbstractMatrix, i::Int) = view(vecs, i, :)

"""Copy one transient patch eigenvector into a dense result slot."""
function _write_result_vector!(
    buffer::_DenseResultBuffer,
    slot::Int,
    source,
    source_index::Int,
    layout::Layout,
)
    copyto!(
        _result_vector_view(buffer.layout, buffer.vecs, slot),
        _result_vector_view(layout, source, source_index),
    )
end

"""Move one already-stored dense result vector to another slot without allocating a matrix."""
function _move_result_vector!(
    buffer::_DenseResultBuffer,
    destination::Int,
    source::Int,
)
    copyto!(
        _result_vector_view(buffer.layout, buffer.vecs, destination),
        _result_vector_view(buffer.layout, buffer.vecs, source),
    )
end

"""
    _result_duplicate_tolerance(eigentol_rescaled)

Choose an eigenvalue comparison tolerance safely above independent patch residual errors.
"""
_result_duplicate_tolerance(eigentol_rescaled::Real) =
    max(8 * abs(Float64(eigentol_rescaled)), 8 * _RESULT_ATOL)

"""
    _is_repeated_pair(buffer, value, vectors, index, layout) -> Bool

Detect a repeated patch copy by requiring both a nearby eigenvalue and that its eigenvector
lies in the span already stored for that eigenvalue cluster. Orthogonal vectors at a genuine
degeneracy are therefore retained.
"""
function _is_repeated_pair(
    buffer::_DenseResultBuffer,
    value::Float64,
    vectors,
    index::Int,
    layout::Layout,
)
    source = _result_vector_view(layout, vectors, index)
    projection = 0.0
    for i in 1:buffer.used
        abs(buffer.vals[i] - value) <= buffer.duplicate_atol || continue
        projection += abs2(dot(_result_vector_view(buffer.layout, buffer.vecs, i), source))
    end
    return projection >= 1 - 1e-4
end

function _is_repeated_pair(
    buffer::_WindowResultBuffer,
    value::Float64,
    vectors,
    index::Int,
    layout::Layout,
)
    source = _result_vector_view(layout, vectors, index)
    projection = 0.0
    for i in eachindex(buffer.vals)
        abs(buffer.vals[i] - value) <= buffer.duplicate_atol || continue
        projection += abs2(dot(buffer.vecs[i], source))
    end
    return projection >= 1 - 1e-4
end

"""Insert one pair into a non-full dense target buffer while preserving eigenvalue order."""
function _insert_sorted_pair!(
    buffer::_DenseResultBuffer,
    value::Float64,
    vectors,
    source_index::Int,
    layout::Layout,
)
    position = searchsortedfirst(view(buffer.vals, 1:buffer.used), value)
    for i in buffer.used:-1:position
        buffer.vals[i + 1] = buffer.vals[i]
        _move_result_vector!(buffer, i + 1, i)
    end
    buffer.used += 1
    buffer.vals[position] = value
    _write_result_vector!(buffer, position, vectors, source_index, layout)
    return nothing
end

"""
    _store_pair!(buffer, value, vectors, source_index, layout)

Store one non-duplicate pair. Dense target buffers retain only their closest `capacity`
pairs; dense full buffers append; dynamic window buffers copy into independent vectors.
"""
function _store_pair!(
    buffer::_DenseResultBuffer,
    value::Real,
    vectors,
    source_index::Int,
    layout::Layout,
)
    λ = Float64(value)
    _is_repeated_pair(buffer, λ, vectors, source_index, layout) && return nothing
    if isnothing(buffer.target)
        buffer.used < length(buffer.vals) ||
            error("More eigenpairs were emitted than the preallocated result can hold.")
        buffer.used += 1
        buffer.vals[buffer.used] = λ
        _write_result_vector!(buffer, buffer.used, vectors, source_index, layout)
        return nothing
    end

    buffer.used < length(buffer.vals) &&
        return _insert_sorted_pair!(buffer, λ, vectors, source_index, layout)

    target = buffer.target::Float64
    farthest = abs(buffer.vals[1] - target) >=
               abs(buffer.vals[buffer.used] - target) ? 1 : buffer.used
    abs(λ - target) < abs(buffer.vals[farthest] - target) || return nothing
    if farthest == 1
        for i in 1:(buffer.used - 1)
            buffer.vals[i] = buffer.vals[i + 1]
            _move_result_vector!(buffer, i, i + 1)
        end
    end
    buffer.used -= 1
    return _insert_sorted_pair!(buffer, λ, vectors, source_index, layout)
end

function _store_pair!(
    buffer::_WindowResultBuffer,
    value::Real,
    vectors,
    source_index::Int,
    layout::Layout,
)
    λ = Float64(value)
    _is_repeated_pair(buffer, λ, vectors, source_index, layout) && return nothing
    vector = similar(_result_vector_view(layout, vectors, source_index))
    copyto!(vector, _result_vector_view(layout, vectors, source_index))
    push!(buffer.vals, λ)
    push!(buffer.vecs, vector)
    return nothing
end

"""
    _store_interval!(buffer, vals, vecs, indices, lower, upper, layout)

Store selected patch pairs inside the requested interval, with numerical endpoint slack.
"""
function _store_interval!(
    buffer,
    vals,
    vecs,
    indices,
    lower::Real,
    upper::Real,
    layout::Layout,
)
    atol = buffer.duplicate_atol
    for i in indices
        value = vals[i]
        lower - atol <= value <= upper + atol &&
            _store_pair!(buffer, value, vecs, i, layout)
    end
    return nothing
end

"""
    _sort_result_buffer!(buffer)

Sort a dense append buffer in place, using one eigenvector scratch allocation rather than
constructing a second result matrix.
"""
function _sort_result_buffer!(buffer::_DenseResultBuffer)
    n = buffer.used
    n < 2 && return nothing
    destinations = invperm(sortperm(view(buffer.vals, 1:n)))
    scratch = similar(_result_vector_view(buffer.layout, buffer.vecs, 1))
    for i in 1:n
        while destinations[i] != i
            j = destinations[i]
            buffer.vals[i], buffer.vals[j] = buffer.vals[j], buffer.vals[i]
            copyto!(scratch, _result_vector_view(buffer.layout, buffer.vecs, i))
            _move_result_vector!(buffer, i, j)
            copyto!(_result_vector_view(buffer.layout, buffer.vecs, j), scratch)
            destinations[i], destinations[j] = destinations[j], destinations[i]
        end
    end
    return nothing
end

"""
    _finalize_result!(buffer) -> (vals, vecs)

Return sorted active storage. A short dense result returns a view of its initialized slots;
a full dense result returns its original matrix.
"""
function _finalize_result!(buffer::_DenseResultBuffer)
    isnothing(buffer.target) && _sort_result_buffer!(buffer)
    n = buffer.used
    vals = collect(view(buffer.vals, 1:n))
    full = n == length(buffer.vals)

    # Internally the buffer may be row-major (the matrix entrypoints run in RowLayout), but the
    # RETURNED eigenvectors are always columns — `vecs[:, i]` is the i-th eigenvector whatever the
    # run used. Row layout pays one transpose here, a real copy rather than a lazy adjoint.
    vecs = buffer.layout isa RowLayout ?
        permutedims(full ? buffer.vecs : view(buffer.vecs, 1:n, :)) :
        (full ? buffer.vecs : view(buffer.vecs, :, 1:n))
    return vals, vecs
end

function _finalize_result!(buffer::_WindowResultBuffer)
    order = sortperm(buffer.vals)
    return buffer.vals[order], buffer.vecs[order]
end
