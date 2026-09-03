# Row-layout operator builders for the matrix entrypoint `polfed(H, …)`. When POLFED owns a concrete
# `H`, it runs Lanczos row-native (the faster layout for the value-grouped kernels), so it needs the
# rescaled transform operator and the map-back as ROW operators (`Y ← X·H`). These helpers build
# them; they are private to `polfed(H, …)`.


@inline function _with_mkl_threads(f::F, nthreads::Integer) where {F}
    nthreads >= 1 ||
        throw(ArgumentError("MKL thread count must be positive"))

    previous = @ccall libmkl_rt.MKL_Set_Num_Threads_Local(
        Cint(nthreads)::Cint
    )::Cint

    try
        return f()
    finally
        @ccall libmkl_rt.MKL_Set_Num_Threads_Local(
            previous::Cint
        )::Cint
    end
end



"""
    _generic_row_rescaled(H, a, b) -> Function

A generic **row**-layout rescaled transform operator `Y ← X·H̃` over a `bs × D` block, where
``\\tilde H = (H - b I)/a`` is formed **once, here**, and each apply is a bare `mul!`. The
`optimize_mapping = false` fallback for the matrix entrypoint's `f!_rescaled` (a matrix not
structured enough for the fast kernels that should still run row-native). Shape-generic — one
closure serves every block size.

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
map-back ([`_row_mapback`](@ref)). For a sparse `H` the copy also gains explicit diagonal entries
wherever the diagonal was structurally zero — that is not overhead but correctness: `H̃ᵢᵢ = -b/a`
is nonzero there.
"""
function _generic_row_rescaled(H::AbstractMatrix, a::Real, b::Real;
                               mkl_threads::Int = Threads.nthreads(:default))
    T = float(eltype(H))
    H̃ = (H - T(b) * I) / T(a)

    return @inline (Y::AbstractMatrix, X::AbstractMatrix) -> begin
        _with_mkl_threads(mkl_threads) do
            mul!(Y, X, H̃)
        end
        return nothing
    end
end


"""
    _row_mapback(H) -> Function

The row-layout map-back operator `Y ← X·H` via `mul!(Y, X, H)`, used as the Lanczos `mapvals`.
Deliberately **generic** (not a value-grouped kernel): map-back is applied only a handful of times
per convergence check, to a *single* strided `1 × D` eigenvector view — where the fast SIMD kernels
give no benefit and the `:shuf` kernel cannot even load (it needs contiguous memory). `mul!` handles
the strided view, any block size, complex, and GPU alike.
"""
_row_mapback(H::AbstractMatrix) =
    @inline (Y::AbstractMatrix, X::AbstractMatrix) -> (mul!(Y, X, H); nothing)

"""
    _build_matrix_rowops(H, choice, bs::Int, a, b, strategy) -> (; f!_rescaled, f!)

Resolve the row-layout operators for the matrix entrypoint from the `optimize_mapping` request
`choice`:

- `f!_rescaled` (the transform, POLFED's bottleneck): the fast value-grouped `bs × D` block kernel
  ([`build_optimized_mapping`](@ref)) when `choice` is truthy, else a generic `mul!` row operator
  ([`_generic_row_rescaled`](@ref));
- `f!` (the one-time map-back `mapvals`): always the generic `mul!` row operator
  ([`_row_mapback`](@ref)), regardless of `choice`.

`bs` is the Krylov block size (runtime); it is lifted to `Val(bs)` here (one-time, at setup) so the
transform kernel closure stays type-stable in the hot loop.
"""
function _build_matrix_rowops(H::AbstractMatrix, choice, bs::Int, a::Real, b::Real,
                              strategy::Parallelization)
    f! = _row_mapback(H)
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
        _generic_row_rescaled(H, a, b; mkl_threads = Threads.nthreads(:default)) :
        build_optimized_mapping(H, choice, Val(bs), a, b).f!_rescaled
    return (; f!_rescaled, f!)
end
