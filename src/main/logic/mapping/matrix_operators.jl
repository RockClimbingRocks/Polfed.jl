# Operator builders for the matrix entrypoints `polfed(H, …)`. When POLFED owns a concrete `H` it
# builds the rescaled transform operator and the map-back for the run's layout: a `ColumnLayout`
# block holds one Krylov vector per column (`Y ← H·X`), a `RowLayout` block one per ROW, and the
# operator applies `H` to each row (`Y ← X·Hᵀ`). These helpers are private to `polfed(H, …)`.


@inline function _with_mkl_threads(f::F, nthreads::Integer) where {F}
    nthreads >= 1 ||
        throw(ArgumentError("MKL thread count must be positive"))
    MKL_jll.is_available() || return f()

    previous = @ccall MKL_jll.libmkl_rt.MKL_Set_Num_Threads_Local(
        Cint(nthreads)::Cint
    )::Cint

    try
        return f()
    finally
        @ccall MKL_jll.libmkl_rt.MKL_Set_Num_Threads_Local(
            previous::Cint
        )::Cint
    end
end


"""
    _AccumulatingOperator

Supertype of the matrix-entrypoint operators that can accumulate, so the transform evaluates them
with [`AxpbyClenshaw`](@ref): one product and one `axpy` per Clenshaw step, with no separate
combine pass. Interface: `op(Y, X)` (`Y ← A·X` per the layout), [`_axpby!`](@ref), [`_add!`](@ref).
CPU: [`_MatrixRescaled`](@ref); GPU: the CUDA extension's cached CUSPARSE operator.
"""
abstract type _AccumulatingOperator <: Function end

"""    _axpby!(op, Y, X, α, β) — `Y ← α·A·X + β·Y` (per `op`'s layout); `β = 0` does not read `Y`."""
function _axpby! end

"""    _add!(op, Y, c, X) — `Y ← Y + c·X`, on the same threads/device as `op`'s product."""
function _add! end


"""
    _generic_rescaled(layout, H, a, b; mkl_threads) -> _MatrixRescaled

A generic rescaled transform operator over a `layout` block, where ``\\tilde H = (H - b I)/a`` is
formed **once, here**, and each apply is a bare `mul!`. The `optimize_mapping = false` operator for
the matrix entrypoint's `f!_rescaled`, and the only CPU one under [`ColumnLayout`](@ref) (the
value-grouped kernels are row kernels). Shape-generic — one operator serves every block size.

# Why prescale instead of rescaling the result
The obvious spelling — `mul!(Y, X, H)` then `@. Y = (Y - b*X)/a` — is correct but pays for the
affine part on **every** apply: a second pass over the block reading `Y`, reading `X` and writing
`Y`, i.e. three block-sized streams of `bs × D` on top of the matvec's own traffic. This is the
POLFED bottleneck: the Clenshaw recurrence calls `f!_rescaled` `K` times per Lanczos matvec, so
that pass is paid `K ×` per iteration.

The shift and the scale are properties of the *operator*, not of the vector, so they can be
folded into the matrix once at setup — exactly what [`rescale`](@ref) does for the value-grouped
kernels (scaling `diags` and each group's value). Then the hot path is a single `mul!` with no
extra traffic at all.

`mkl_threads` is how many threads each `mul!` may use, and it is the CALLER's decision, not the
operator's. Under [`VectorWise`](@ref) the strategy already opens one task per Krylov vector, and
each task then enters this closure: leaving the default would have every task ask MKL for the
full thread count, `nt x nt` in total. Measured at `nt = 8`, that oversubscription cost **2.85x**
against the same arithmetic issued as one block `mul!`. `MKL_Set_Num_Threads_Local` is
thread-local, so a `BLAS.set_num_threads(1)` outside the region does not restrain it.

The cost is one rescaled copy of `H` alongside the original, which is still needed for the
map-back ([`_mapback`](@ref)). For a sparse `H` the copy also gains explicit diagonal entries
wherever the diagonal was structurally zero — that is not overhead but correctness: `H̃ᵢᵢ = -b/a`
is nonzero there.
"""
function _generic_rescaled(layout::Layout, H::AbstractMatrix, a::Real, b::Real;
                           mkl_threads::Int = Threads.nthreads(:default))
    T = float(eltype(H))
    H̃ = (H - T(b) * I) / T(a)
    return _MatrixRescaled(layout, layout isa RowLayout ? _row_operand(H̃) : H̃, mkl_threads)
end

"""
    _row_operand(A) -> AbstractMatrix

The right factor `M` with `X·M = X·Aᵀ`, i.e. `A` applied to every row of `X`: `A` itself when it is
real (a real Hermitian `A` is symmetric), `transpose(A)` materialized when it is complex — there
`Aᵀ = conj(A)`, and `X·A` would return the CONJUGATE eigenvectors.
"""
_row_operand(A::AbstractMatrix) = eltype(A) <: Real ? A : copy(transpose(A))

"""
    _MatrixRescaled(layout, M, mkl_threads)

The generic CPU operator built by [`_generic_rescaled`](@ref): callable as `op(Y, X)` — `Y ← M·X`
under [`ColumnLayout`](@ref), `Y ← X·M` under [`RowLayout`](@ref), where `M` is `H̃` or its
[`_row_operand`](@ref) — plus the accumulating forms the transform's [`AxpbyClenshaw`](@ref) uses,
so a whole Clenshaw step stays inside MKL's thread team instead of alternating with a Julia
threaded combine.
"""
struct _MatrixRescaled{L<:Layout,M<:AbstractMatrix} <: _AccumulatingOperator
    layout::L
    M::M
    mkl_threads::Int
end

"""    _matmul!(layout, Y, M, X, α, β) — `Y ← α·M·X + β·Y` ([`ColumnLayout`](@ref)) or `Y ← α·X·M + β·Y` ([`RowLayout`](@ref))."""
@inline _matmul!(::ColumnLayout, Y, M, X, α, β) = mul!(Y, M, X, α, β)
@inline _matmul!(::RowLayout,    Y, M, X, α, β) = mul!(Y, X, M, α, β)

function (op::_MatrixRescaled)(Y::AbstractVecOrMat, X::AbstractVecOrMat)
    _with_mkl_threads(() -> _matmul!(op.layout, Y, op.M, X, true, false), op.mkl_threads)
    return nothing
end

function _axpby!(op::_MatrixRescaled, Y::AbstractVecOrMat, X::AbstractVecOrMat, α, β)
    _with_mkl_threads(() -> _matmul!(op.layout, Y, op.M, X, α, β), op.mkl_threads)
    return nothing
end

function _add!(op::_MatrixRescaled, Y::AbstractVecOrMat, c, X::AbstractVecOrMat)
    _with_mkl_threads(() -> _mkl_axpy!(c, X, Y), op.mkl_threads)
    return nothing
end

"""
    _mkl_axpy!(c, X, Y) -> Y

`Y ← Y + c·X` through MKL's own `cblas_?axpy_64`, whatever BLAS Julia is using: with OpenBLAS
behind `LinearAlgebra`, `axpy!` would start a third thread pool next to MKL's. Falls back to
`axpy!` for anything but contiguous CPU arrays of a BLAS type, and where MKL is not available.
"""
function _mkl_axpy!(c, X::Array{T}, Y::Array{T}) where {T<:LinearAlgebra.BlasFloat}
    length(X) == length(Y) || throw(DimensionMismatch("axpy: X has $(length(X)) entries, Y $(length(Y))."))
    MKL_jll.is_available() || return axpy!(c, X, Y)
    _cblas_axpy!(Int64(length(X)), T(c), X, Y)
    return Y
end
_mkl_axpy!(c, X, Y) = axpy!(c, X, Y)

for (T, fn) in ((Float64, :cblas_daxpy_64), (Float32, :cblas_saxpy_64))
    @eval _cblas_axpy!(n::Int64, a::$T, X::Array{$T}, Y::Array{$T}) =
        @ccall MKL_jll.libmkl_rt.$fn(n::Int64, a::$T, X::Ptr{$T}, 1::Int64, Y::Ptr{$T}, 1::Int64)::Cvoid
end
for (T, fn) in ((ComplexF64, :cblas_zaxpy_64), (ComplexF32, :cblas_caxpy_64))
    @eval _cblas_axpy!(n::Int64, a::$T, X::Array{$T}, Y::Array{$T}) =
        @ccall MKL_jll.libmkl_rt.$fn(n::Int64, Ref(a)::Ref{$T}, X::Ptr{$T}, 1::Int64, Y::Ptr{$T},
                                     1::Int64)::Cvoid
end


"""
    _mapback(layout, H) -> Function

The map-back operator applying `H` to each Krylov vector of a `layout` block (`Y ← H·X` /
`Y ← X·Hᵀ`), used as the Lanczos `mapvals`. Deliberately **generic** (not a value-grouped kernel):
map-back is applied only a handful of times per convergence check, to a *single* `1 × D` (row) or
length-`D` (column) eigenvector view — where the fast SIMD kernels give no benefit and the `:shuf`
kernel cannot even load (it needs contiguous memory). `mul!` handles the strided view, any block
size, and complex. A complex `H` takes a lazy `transpose` here (no copy; see [`_row_operand`](@ref)).
"""
_mapback(::ColumnLayout, H::AbstractMatrix) =
    @inline (Y::AbstractVecOrMat, X::AbstractVecOrMat) -> (mul!(Y, H, X); nothing)
function _mapback(::RowLayout, H::AbstractMatrix)
    M = eltype(H) <: Real ? H : transpose(H)
    return @inline (Y::AbstractMatrix, X::AbstractMatrix) -> (mul!(Y, X, M); nothing)
end

"""
    _gpu_matrix_operator(A, layout) -> _AccumulatingOperator

The device operator applying `A` to each Krylov vector of a `layout` block — cached CUSPARSE
SpMV/SpMM, defined by the CUDA extension. Without CUDA.jl loaded a device matrix cannot reach here.
"""
_gpu_matrix_operator(A, ::Layout) = _cuda_unavailable()

"""
    _build_matrix_ops(H, choice, bs::Int, a, b, strategy, layout) -> (; f!_rescaled, f!)

Resolve the matrix entrypoint's operators for `layout` from the `optimize_mapping` request `choice`:

- `f!_rescaled` (the transform, POLFED's bottleneck): on the CPU the fast value-grouped `bs × D`
  block kernel ([`build_optimized_mapping`](@ref), row layout only) when `choice` is truthy, else a
  generic `mul!` operator ([`_generic_rescaled`](@ref)); on the GPU the extension's cached CUSPARSE
  operator ([`_gpu_matrix_operator`](@ref)) over `H̃`;
- `f!` (the one-time map-back `mapvals`): the generic operator ([`_mapback`](@ref)) on the CPU, the
  same CUSPARSE operator over `H` on the GPU, regardless of `choice`.

`bs` is the Krylov block size (runtime); it is lifted to `Val(bs)` here (one-time, at setup) so the
transform kernel closure stays type-stable in the hot loop.

# Throws
- `ArgumentError`: `choice` requests the value-grouped kernels under [`ColumnLayout`](@ref) or on
  the GPU, where they do not exist.
"""
function _build_matrix_ops(H::AbstractMatrix, choice, bs::Int, a::Real, b::Real,
                           strategy::Parallelization, layout::Layout)
    on_gpu = is_gpu_sparse_matrix(H) || is_gpu_array(H)
    (choice === false || (!on_gpu && layout isa RowLayout)) || throw(ArgumentError(
        "optimize_mapping = $(repr(choice)) asks for the value-grouped kernels, which are CPU " *
        "row-layout kernels; this run is $(on_gpu ? "on the GPU" : "column-layout"). Use " *
        "optimize_mapping = false$(on_gpu ? "" : ", or MappingConfig(layout = RowLayout())")."))
    if on_gpu
        T = float(eltype(H))
        return (; f!_rescaled = _gpu_matrix_operator((H - T(b) * I) / T(a), layout),
                  f! = _gpu_matrix_operator(H, layout))
    end
    f! = _mapback(layout, H)
    # `f!_rescaled` is WHOLE-BLOCK, always: it is the TRANSFORM operator, applied to the
    # `bs x D` Krylov block on every Clenshaw step. It has exactly one consumer — the DoS stage
    # builds its own generic operator (`dos_operator`) and probes one vector at a time, so the
    # density no longer depends on the Krylov block size.
    #
    # Building this one per-vector for a VectorWise run — `Val(1)` — is still not a local choice:
    # measured, a `Val(1)` kernel handed a `bs x D` block writes row 1 and leaves the rest zero,
    # 93% relative error. A per-vector transform has to be a SECOND operator, not a replacement
    # for this one.
    f!_rescaled = choice === false ?
        _generic_rescaled(layout, H, a, b; mkl_threads = Threads.nthreads(:default)) :
        build_optimized_mapping(H, choice, Val(bs), a, b).f!_rescaled
    return (; f!_rescaled, f!)
end
