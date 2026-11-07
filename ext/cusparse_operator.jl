# The device operator behind the matrix entrypoints on the GPU: `A` applied to every Krylov vector
# of a column (`Y ← A·X`) or row (`Y ← X·Aᵀ`) block through CUSPARSE, with the descriptors'
# workspace sized, allocated and preprocessed once per block shape instead of on every call.

using CUDA.CUSPARSE: CuSparseMatrixDescriptor, handle,
                     cusparseSpMM, cusparseSpMM_bufferSize, cusparseSpMM_preprocess,
                     cusparseCreateDnMat, cusparseDestroyDnMat, cusparseDnMatDescr_t,
                     CUSPARSE_SPMM_ALG_DEFAULT, CUSPARSE_SPMM_CSR_ALG2, CUSPARSE_SPMM_CSR_ALG3

"""
    _row_alg2_from(T) -> Int

Row-layout blocks narrower than this use `CUSPARSE_SPMM_CSR_ALG3`, wider ones `ALG2`. Measured on
a GH200 (XXZ L = 22, per vector): ALG3 is 1.5–2× faster at `bs = 8–16` for every type; at
`bs = 32` ALG2 wins for Float64 (22.5 vs 25.3 µs) but loses for Float32 (20.1 vs 10.6) and
ComplexF64 (60.9 vs 53.0); at `bs = 64` ALG2 wins or ties for all three.
"""
_row_alg2_from(::Type{Float64}) = 32
_row_alg2_from(::Type) = 64

"""
    _DnMat(X, rowmajor)

A CUSPARSE dense-matrix descriptor over any strided device matrix with unit first stride (CUDA.jl's
own wants a dense one). `rowmajor` describes the column-major `bs × D` block `X` as the row-major
`D × bs` matrix `Xᵀ`, with leading dimension `stride(X, 2)` — which also covers a strided `1 × D`
row view of a larger block.
"""
mutable struct _DnMat
    handle::cusparseDnMatDescr_t
    function _DnMat(X::CUDA.StridedCuMatrix, rowmajor::Bool)
        stride(X, 1) == 1 || throw(ArgumentError("CUSPARSE needs a unit first stride, got $(strides(X))"))
        rows, cols = rowmajor ? reverse(size(X)) : size(X)
        ref = Ref{cusparseDnMatDescr_t}()
        cusparseCreateDnMat(ref, rows, cols, max(stride(X, 2), rowmajor ? cols : rows), pointer(X),
                            eltype(X), rowmajor ? 'R' : 'C')
        obj = new(ref[])
        finalizer(d -> cusparseDestroyDnMat(d.handle), obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cusparseDnMatDescr_t}, d::_DnMat) = d.handle

"""
    CuSparseOperator(A, layout)

`A` (CSR) applied to every Krylov vector of a `layout` block: `Y ← A·X` ([`Polfed.ColumnLayout`](@ref))
or `Y ← X·Aᵀ` ([`Polfed.RowLayout`](@ref) — row-major SpMM over the same CSR, so a complex Hermitian
`A` needs no conjugated copy). A single contiguous vector runs as a `D × 1` column block (faster
than SpMV). Implements the [`Polfed._AccumulatingOperator`](@ref) interface, so the transform runs
it under [`Polfed.AxpbyClenshaw`](@ref).

Workspaces are cached per (stream, block size, leading dimension, algorithm) and
preprocessed once: CUSPARSE's own `mul!` re-sizes and re-preprocesses on every call, which costs
~5 µs — 40% of a `D = 5·10⁴` product. Keying by stream keeps concurrent tasks (each has its own
stream) off each other's buffers.
"""
struct CuSparseOperator{T,M<:CuSparseMatrixCSR{T},L<:Polfed.Layout} <: Polfed._AccumulatingOperator
    A::M
    layout::L
    descA::CuSparseMatrixDescriptor
    workspaces::Dict{Tuple{UInt,Int,Int,Int},CuVector{UInt8}}
    lock::ReentrantLock
end
CuSparseOperator(A::CuSparseMatrixCSR{T}, layout::Polfed.Layout) where {T} =
    CuSparseOperator{T,typeof(A),typeof(layout)}(A, layout, CuSparseMatrixDescriptor(A, 'O'),
                                                 Dict{Tuple{UInt,Int,Int,Int},CuVector{UInt8}}(),
                                                 ReentrantLock())

Polfed._gpu_matrix_operator(A::CuSparseMatrixCSR, layout::Polfed.Layout) = CuSparseOperator(A, layout)
Polfed._gpu_matrix_operator(A::Union{CuSparseMatrixCSC,CuSparseMatrixCOO}, layout::Polfed.Layout) =
    CuSparseOperator(CuSparseMatrixCSR(A), layout)

(op::CuSparseOperator)(Y::AbstractVecOrMat, X::AbstractVecOrMat) =
    Polfed._axpby!(op, Y, X, true, false)

Polfed._add!(::CuSparseOperator, Y::AbstractVecOrMat, c, X::AbstractVecOrMat) = (axpy!(c, X, Y); nothing)

"""
    _workspace!(make, op, key) -> CuVector{UInt8}

The cached CUSPARSE workspace for `key` on the current task's stream, built by `make()` on first
use. Only the lookup holds `op.lock`; the product runs outside it.
"""
function _workspace!(make, op::CuSparseOperator, key)
    lock(op.lock) do
        get!(make, op.workspaces, (objectid(CUDA.stream()), key...))
    end
end

"""
    _single_vector(layout, X) -> Union{CuVector,Nothing}

`X` as a plain device vector when it holds exactly one contiguous vector — a vector, a `D × 1`
block, or a contiguous `1 × D` row — else `nothing`. Such a block runs as the column-major `D × 1`
block it is in memory.
"""
# On a GH200 that `D × 1` SpMM is 5–33% faster than SpMV (XXZ L = 18–24, QSun;
# Float32/Float64/ComplexF64) and faster than a one-row row-major SpMM.
_single_vector(::Polfed.Layout, X::CUDA.DenseCuVector) = X
_single_vector(::Polfed.ColumnLayout, X::CUDA.StridedCuMatrix) =
    size(X, 2) == 1 && stride(X, 1) == 1 ? vec(X) : nothing
_single_vector(::Polfed.RowLayout, X::CUDA.StridedCuMatrix) =
    size(X, 1) == 1 && stride(X, 2) == 1 ? vec(X) : nothing

function Polfed._axpby!(op::CuSparseOperator, Y::AbstractVecOrMat, X::AbstractVecOrMat, α, β)
    x = _single_vector(op.layout, X)
    x === nothing ||
        return _spmm!(op, reshape(_single_vector(op.layout, Y), :, 1), reshape(x, :, 1), false, α, β)
    return _spmm!(op, Y, X, op.layout isa Polfed.RowLayout, α, β)
end

"""
    _spmm!(op, Y, X, rowmajor, α, β)

`Y ← α·A·X + β·Y` for a column-major block, or (`rowmajor`) its row-major reading
`Y ← α·X·Aᵀ + β·Y`, through `cusparseSpMM` with a cached, preprocessed workspace. Assumes `Y` has
the shape and strides of `X` (the workspace is keyed by `X` alone).
"""
function _spmm!(op::CuSparseOperator{T}, Y::AbstractMatrix, X::AbstractMatrix, rowmajor::Bool,
                α, β) where {T}
    bs = rowmajor ? size(X, 1) : size(X, 2)
    alg = !rowmajor ? CUSPARSE_SPMM_ALG_DEFAULT :
          bs < _row_alg2_from(T) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2
    descX, descY = _DnMat(X, rowmajor), _DnMat(Y, rowmajor)
    a, b = Ref{T}(α), Ref{T}(β)
    buf = _workspace!(op, (bs, stride(X, 2), Int(alg))) do
        sz = Ref{Csize_t}(1000)
        cusparseSpMM_bufferSize(handle(), 'N', 'N', a, op.descA, descX, b, descY, T, alg, sz)
        w = CuVector{UInt8}(undef, max(Int(sz[]), 1))
        try    # an analysis pass for the algorithms that have one; the others report unsupported
            cusparseSpMM_preprocess(handle(), 'N', 'N', a, op.descA, descX, b, descY, T, alg, w)
        catch err
            err isa CUSPARSE.CUSPARSEError || rethrow()
        end
        w
    end
    cusparseSpMM(handle(), 'N', 'N', a, op.descA, descX, b, descY, T, alg, buf)
    return nothing
end

"""
    CuDenseOperator(A, layout)

A dense device matrix applied to every Krylov vector of a `layout` block (CUBLAS GEMM): `Y ← A·X` or
`Y ← X·Aᵀ`. The dense counterpart of [`CuSparseOperator`](@ref).
"""
struct CuDenseOperator{M<:CuMatrix,L<:Polfed.Layout} <: Polfed._AccumulatingOperator
    A::M
    layout::L
end
Polfed._gpu_matrix_operator(A::CuMatrix, layout::Polfed.Layout) = CuDenseOperator(A, layout)

(op::CuDenseOperator)(Y::AbstractVecOrMat, X::AbstractVecOrMat) = Polfed._axpby!(op, Y, X, true, false)
Polfed._axpby!(op::CuDenseOperator{<:Any,Polfed.ColumnLayout}, Y::AbstractVecOrMat,
               X::AbstractVecOrMat, α, β) = (mul!(Y, op.A, X, α, β); nothing)
Polfed._axpby!(op::CuDenseOperator{<:Any,Polfed.RowLayout}, Y::AbstractMatrix, X::AbstractMatrix,
               α, β) = (mul!(Y, X, transpose(op.A), α, β); nothing)
Polfed._add!(::CuDenseOperator, Y::AbstractVecOrMat, c, X::AbstractVecOrMat) = (axpy!(c, X, Y); nothing)
