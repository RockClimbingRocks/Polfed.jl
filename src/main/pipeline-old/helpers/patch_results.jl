function _find_left_overlap(
    vals::AbstractVector{T},
    vals_i::AbstractVector{T},
    atol::Real,
) where {T<:Number}
    isempty(vals_i) && throw(AssertionError("_find_left_overlap: vals_i is empty."))
    isempty(vals) && return Int[]

    start = findlast(val -> isapprox(val, first(vals_i); atol=atol, rtol=0), vals)
    start === nothing && throw(ErrorException("No overlap found."))

    overlap_length = lastindex(vals) - start + 1
    length(vals_i) >= overlap_length || throw(ErrorException("Incomplete overlap."))

    for offset in 0:(overlap_length - 1)
        isapprox(vals[start + offset], vals_i[firstindex(vals_i) + offset]; atol=atol, rtol=0) ||
            throw(ErrorException("Overlap values do not match."))
    end

    return collect(start:lastindex(vals))
end


function _find_right_overlap(
    vals::AbstractVector{T},
    vals_i::AbstractVector{T},
    atol::Real,
) where {T<:Number}
    isempty(vals_i) && throw(AssertionError("_find_right_overlap: vals_i is empty."))
    isempty(vals) && return Int[]

    stop = findfirst(val -> isapprox(val, last(vals_i); atol=atol, rtol=0), vals)
    stop === nothing && throw(ErrorException("No overlap found."))

    overlap_length = stop - firstindex(vals) + 1
    length(vals_i) >= overlap_length || throw(ErrorException("Incomplete overlap."))
    incoming_start = lastindex(vals_i) - overlap_length + 1

    for offset in 0:(overlap_length - 1)
        isapprox(vals[firstindex(vals) + offset], vals_i[incoming_start + offset]; atol=atol, rtol=0) ||
            throw(ErrorException("Overlap values do not match."))
    end

    return collect(firstindex(vals):stop)
end


_number_of_vectors(::ColumnLayout, vecs::AbstractMatrix) = size(vecs, 2)
_number_of_vectors(::RowLayout, vecs::AbstractMatrix) = size(vecs, 1)
_eigenvector_view(::ColumnLayout, vecs::AbstractMatrix, i) = view(vecs, :, i)
_eigenvector_view(::RowLayout, vecs::AbstractMatrix, i) = view(vecs, i, :)


"""
    patch_results_sequential!(vals_i, vecs_i, vals, vecs, atol, layout=ColumnLayout())

Append the non-overlapping part of a Lanczos patch. Existing eigenvalues and
eigenvectors are never rebuilt. New eigenvectors are stored as views of `vecs_i`.
"""
function patch_results_sequential!(
    vals_i::AbstractVector{T},
    vecs_i::AbstractMatrix{V},
    vals::Vector{T},
    vecs::Vector{W},
    atol::Real,
    layout::Layout=ColumnLayout(),
) where {T<:Real,V<:Number,W<:AbstractVector{V}}
    _number_of_vectors(layout, vecs_i) == length(vals_i) ||
        throw(DimensionMismatch("Each eigenvalue must have one eigenvector."))
    length(vals) == length(vecs) ||
        throw(DimensionMismatch("Accumulated eigenvalues and eigenvectors do not match."))

    overlap = _find_left_overlap(vals, vals_i, atol)
    new_indices = (length(overlap) + 1):length(vals_i)

    append!(vals, @view vals_i[new_indices])
    for i in new_indices
        push!(vecs, _eigenvector_view(layout, vecs_i, i))
    end
    return nothing
end


"""
    patch_results_owned!(vals_i, vecs_i, own, vals, vecs, layout=ColumnLayout();
                         include_right=false)

Append the eigenpairs a patch **owns** onto the running `vals`/`vecs`. A patch converges its
(padded) window but owns only its unpadded interval `own`; this keeps exactly the eigenpairs
whose **rescaled** value lies in `own`, half-open `[own.left, own.right)`. Because the unpadded
patch intervals **tile** the requested window with bit-exact shared endpoints, half-open
ownership assigns every eigenvalue to exactly one patch — no gaps, no duplicates. The final patch
passes `include_right=true` so the window's closed right endpoint is kept.

This is the robust alternative to [`patch_results_sequential!`](@ref): ownership is decided by
VALUE, not by matching a shared prefix, so it is immune to the [`Window`](@ref) brackets and the
per-patch count differences that make strict prefix-alignment misalign. Every owned eigenvalue is
converged well inside the patch's padded window (the padding keeps it clear of the fragile bracket
edges). Assumes `vals_i` is sorted ascending (as `lanczos` returns); with patches processed
left-to-right the accumulated `vals` stays ascending. New eigenvectors are stored as views of
`vecs_i`.

The two boundary flags control whether each endpoint is **closed** (owned) or **open** (left to a
neighbour). The default `[own.left, own.right)` tiles cleanly for adjacent interior patches; the
last patch of a window sets `include_right=true`, and a patch abutting an already-harvested set on
its left (e.g. the first interior patch after an edge peel, whose lower boundary the edge patch
already owns) sets `include_left=false`.

# Arguments
- `vals_i`, `vecs_i`: one patch's converged eigenvalues (rescaled, ascending) and eigenvectors.
- `own::Interval`: the patch's unpadded (ownership) interval, in rescaled coordinates.
- `vals`, `vecs`: the running accumulators, appended in place.
- `include_left::Bool=true`: keep the closed left endpoint (set `false` to leave it to a neighbour).
- `include_right::Bool=false`: keep the closed right endpoint (`true` for a window's last patch).
"""
function patch_results_owned!(
    vals_i::AbstractVector{T},
    vecs_i::AbstractMatrix{V},
    own::Interval,
    vals::Vector{T},
    vecs::Vector{W},
    layout::Layout=ColumnLayout();
    include_left::Bool=true,
    include_right::Bool=false,
) where {T<:Real,V<:Number,W<:AbstractVector{V}}
    _number_of_vectors(layout, vecs_i) == length(vals_i) ||
        throw(DimensionMismatch("Each eigenvalue must have one eigenvector."))
    length(vals) == length(vecs) ||
        throw(DimensionMismatch("Accumulated eigenvalues and eigenvectors do not match."))

    for k in eachindex(vals_i)
        v = vals_i[k]
        ge_left = include_left  ? (own.left  <= v) : (own.left  < v)
        le_right = include_right ? (v <= own.right) : (v < own.right)
        (ge_left && le_right) || continue
        push!(vals, v)
        push!(vecs, _eigenvector_view(layout, vecs_i, k))
    end
    return nothing
end
