"""
    orthogonalization!(factorization::KrylovFactorization) -> nothing

Orthogonalize the residual workspace against the latest basis vectors/blocks using
the current overlap and norm coefficients, then repair the rounding error that
subtraction leaves behind ([`dgks_correction!`](@ref)).
"""
@inline function orthogonalization!(factorization::KrylovFactorization)
    v1 = factorization.v_last
    v2 = factorization.v_secondlast
    α = _device_coeff(factorization, getoverlap(factorization), 1)
    β = _device_coeff(factorization, getnorm(factorization), 2)
    orthogonalization!(factorization.layout, factorization.r, v1, v2, α, β)
    dgks_correction!(factorization)
    return nothing
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


"""
    dgks_correction!(factorization) -> nothing

Second projection against the two most recent basis blocks, with coefficients
re-measured from the already-subtracted residual:

    δα = V_j' R₁ ,   R₂ = R₁ − V_j δα ,   α_j ← α_j + δα
    δβ = V_{j-1}' R₂ ,   R₃ = R₂ − V_{j-1} δβ

The recurrence in [`orthogonalization!`](@ref) removes a component of size `O(‖A·V_j‖)`
and keeps a residual of size `‖β_{j+1}‖`, so its cancellation leaves an orthogonality
defect of `ε·‖A·V_j‖/‖β_{j+1}‖` — growing as the filter converges and `β` collapses.
Those two blocks are exactly the ones [`reorthogonalization!`](@ref) skips via
`all_withoutlasttwo`, so nothing else repairs them. The second pass has no cancellation
(`‖R₂‖ ≈ ‖R₁‖`), so the defect drops to `O(ε)` unconditionally.

`δα` is accumulated into the stored overlap so `T[j,j]` remains the Rayleigh quotient
of the basis actually held. `δβ` is not stored: it is `O(ε‖A‖)`, the same order as the
arithmetic, and `β` already lives in two mirrored views of `fact.mat`.
"""
@inline function dgks_correction!(fact::BlockLanczosFactorization)
    layout, r, δ = fact.layout, fact.r, fact.orth_coeff
    overlap!(layout, δ, fact.v_last, r)
    project!(layout, r, fact.v_last, δ)
    addoverlap!(fact, to_host(δ))
    if fact.krylovdim >= 2 * fact.blocksize          # mirrors the getnorm guard
        overlap!(layout, δ, fact.v_secondlast, r)
        project!(layout, r, fact.v_secondlast, δ)
    end
    return nothing
end

@inline function dgks_correction!(fact::LanczosFactorization)
    r, v1, v2 = fact.r, fact.v_last, fact.v_secondlast
    δα = dot(v1, r)
    axpy!(-δα, v1, r)
    addoverlap!(fact, real(δα))
    fact.krylovdim >= 2 && axpy!(-dot(v2, r), v2, r)
    return nothing
end