"""
    orthogonalization!(factorization::KrylovFactorization) -> nothing

Orthogonalize the residual workspace against the latest basis vectors/blocks using
the current overlap and norm coefficients.
"""
@inline function orthogonalization!(factorization::KrylovFactorization)
    v1 = factorization.v_last
    v2 = factorization.v_secondlast
    α = getoverlap(factorization)
    β = getnorm(factorization)
    orthogonalization!(factorization.layout, factorization.r, v1, v2, α, β)
end

"""
    orthogonalization!(layout, r, v1, v2, α, β) -> nothing

Low-level orthogonalization kernels for vector and block variants. `r` is mutated
in place. The block variant subtracts both couplings via [`project!`](@ref), so the side
of each multiplication follows `layout`.
"""
@inline function orthogonalization!(::Layout, r::AbstractVector, v1::AbstractVector, v2::AbstractVector, α::Real, β::Real)
    axpy!(-α, v1, r)
    axpy!(-β, v2, r)
    return r
end

@inline function orthogonalization!(layout::Layout, r::AbstractMatrix, v1::AbstractMatrix, v2::AbstractMatrix, α::AbstractMatrix, β::AbstractMatrix)
    project!(layout, r, v1, α)                                    # r ← r − v1·α (col) / α·v1 (row)
    # `β = getnorm` is the below-diagonal coupling T[j,j-1]: column stores R (so the recurrence
    # uses β'), row stores L' (so it uses β itself) — see tridiagonalization! storage.
    project!(layout, r, v2, layout isa RowLayout ? β : β')        # r ← r − v2·β' (col) / β·v2 (row)
    return r
end

"""Mixed vector/block overload of `orthogonalization!` (column-only edge case)."""
@inline function orthogonalization!(::Layout, r::AbstractMatrix, v1::AbstractVector, v2::AbstractVector, α::AbstractMatrix, β::AbstractMatrix)
    size(r, 2) == 1 || throw(DimensionMismatch("vector basis requires a one-column residual"))
    rv = view(r, :, 1)
    # The one-element coefficient access must also be legal for device matrices.
    gpu_allowscalar() do
        axpy!(-α[1, 1], v1, rv)
        axpy!(-conj(β[1, 1]), v2, rv)
    end
    return r
end
