# Shared helpers for the row-layout CPU kernels: thread chunking + SVector block load/store.
# Row layout means the block `X`, `Y` are `bs × D`; the block index `b` is the contiguous
# (fast) dimension, so a whole column `X[:, j]` is `bs` contiguous elements. Ported from the
# test-10-fable benchmark kernels.

"""
    _chunks_aligned(D::Int, align::Int) -> (nt, chunk)

Split `D` rows across `Threads.nthreads()` threads into chunks whose length is a multiple of
`align` (so every chunk start is `align`-aligned, required by the `G`-way-ILP inner blocks).
"""
@inline function _chunks_aligned(D::Int, align::Int)
    nt = Threads.nthreads()
    chunk = align * cld(cld(D, nt), align)
    return nt, chunk
end

"""
    loadv(X, j, ::Val{bs}, ::Type{T}) -> SVector{bs,T}

Load column `j` of the row-layout block `X` (`bs × D`) as a stack-allocated `SVector{bs}`.
"""
@inline loadv(X, j, ::Val{bs}, ::Type{T}) where {bs,T} =
    SVector{bs,T}(ntuple(b -> @inbounds(X[b, j]), Val(bs)))

"""
    storev!(Y, i, acc::SVector{bs}) -> nothing

Store the accumulator `acc` into column `i` of the row-layout block `Y` (`bs × D`).
"""
@inline function storev!(Y, i, acc::SVector{bs}) where {bs}
    @inbounds for b in 1:bs
        Y[b, i] = acc[b]
    end
    return nothing
end
