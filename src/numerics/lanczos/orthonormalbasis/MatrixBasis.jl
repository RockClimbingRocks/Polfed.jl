"""
    MatrixBasis{T,E,L}(maxdim::Int, x0::AbstractVecOrMat, pu::ProcessingUnit, layout::Layout=ColumnLayout())

Dense basis storage for Lanczos vectors/blocks. `layout` fixes the storage orientation:
[`ColumnLayout`](@ref) stores the `nvecs` Krylov vectors as **columns** (`D × maxdim`),
[`RowLayout`](@ref) as **rows** (`maxdim × D`). All orientation-sensitive access
(`add!`/`allvecs`/`all_withoutlasttwo`) dispatches on the `layout` type parameter.
"""
mutable struct MatrixBasis{T,E,L<:Layout} <: OrthonormalBasis
    basis::AbstractMatrix{E}
    nvecs::Int
    blocksize::Int
    layout::L

    function MatrixBasis{E}(maxdim::Int, x0::T, pu::ProcessingUnit,
                            layout::L=ColumnLayout()) where {T<:AbstractVecOrMat, E<:Number, L<:Layout}
        dim = opdim(layout, x0)
        bs = blocksize(layout, x0)
        basis = layout isa RowLayout ? pu.Matrix{E}(undef, maxdim, dim) : pu.Matrix{E}(undef, dim, maxdim)
        new{T,E,L}(basis, 0, bs, layout)
    end
end

# --- ColumnLayout: vectors stored as columns (D × maxdim) --------------------

"""
    add!(basis::MatrixBasis, v::AbstractArray) -> nothing

Append one vector or block to `basis`, shifting the oldest vectors out when capacity is
exceeded. Vectors are columns ([`ColumnLayout`](@ref)) or rows ([`RowLayout`](@ref)).
"""
function add!(basis::MatrixBasis{T,E,ColumnLayout}, v::AbstractArray) where {T,E}
    ncols = ndims(v) == 1 ? 1 : size(v, 2)
    vmat = ndims(v) == 1 ? reshape(v, :, 1) : v

    maxcols = size(basis.basis, 2)

    if basis.nvecs + ncols <= maxcols
        basis.basis[:, basis.nvecs+1 : basis.nvecs+ncols] .= vmat
        basis.nvecs += ncols
    else
        shift = basis.nvecs + ncols - maxcols
        for i in 1:(basis.nvecs - shift)
            basis.basis[:, i] .= basis.basis[:, i + shift]
        end
        basis.basis[:, basis.nvecs - shift + 1 : basis.nvecs - shift + ncols] .= vmat
        basis.nvecs = maxcols
    end

    return nothing
end

"""Return a view of all stored basis vectors: columns (`D × nvecs`) or rows (`nvecs × D`)."""
function allvecs(basis::MatrixBasis{T,E,ColumnLayout}) where {T,E}
    basis.nvecs == 0 && throw(ArgumentError("Basis is empty"))
    return view(basis.basis, :, 1:basis.nvecs)
end

"""Return a view excluding the most recent two blocks (used for reorthogonalization)."""
function all_withoutlasttwo(basis::MatrixBasis{T,E,ColumnLayout}) where {T,E}
    blocksize = basis.blocksize
    basis.nvecs < 2 * blocksize && throw(ArgumentError("Not enough vectors in the basis"))
    return view(basis.basis, :, 1:basis.nvecs-2blocksize)
end

# --- RowLayout: vectors stored as rows (maxdim × D) --------------------------

function add!(basis::MatrixBasis{T,E,RowLayout}, v::AbstractArray) where {T,E}
    nrows = ndims(v) == 1 ? 1 : size(v, 1)
    vmat = ndims(v) == 1 ? reshape(v, 1, :) : v

    maxrows = size(basis.basis, 1)

    if basis.nvecs + nrows <= maxrows
        basis.basis[basis.nvecs+1 : basis.nvecs+nrows, :] .= vmat
        basis.nvecs += nrows
    else
        shift = basis.nvecs + nrows - maxrows
        for i in 1:(basis.nvecs - shift)
            basis.basis[i, :] .= basis.basis[i + shift, :]
        end
        basis.basis[basis.nvecs - shift + 1 : basis.nvecs - shift + nrows, :] .= vmat
        basis.nvecs = maxrows
    end

    return nothing
end

function allvecs(basis::MatrixBasis{T,E,RowLayout}) where {T,E}
    basis.nvecs == 0 && throw(ArgumentError("Basis is empty"))
    return view(basis.basis, 1:basis.nvecs, :)
end

function all_withoutlasttwo(basis::MatrixBasis{T,E,RowLayout}) where {T,E}
    blocksize = basis.blocksize
    basis.nvecs < 2 * blocksize && throw(ArgumentError("Not enough vectors in the basis"))
    return view(basis.basis, 1:basis.nvecs-2blocksize, :)
end
