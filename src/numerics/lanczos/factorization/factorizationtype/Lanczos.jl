"""
    LanczosFactorization(maxdim::Int, basis::OrthonormalBasis, x0::AbstractVector{T}, pu::ProcessingUnit)

State container for scalar (non-block) Lanczos factorization.
"""
mutable struct LanczosFactorization{S<:Real,T<:Number,P,L<:Layout} <: KrylovFactorization{S,T,P}
    krylovdim::Int                 # current Krylov dimension
    basis::OrthonormalBasis        # basis of length krylovdim
    r::AbstractVector{T}
    v_last::AbstractVector{T}
    v_secondlast::AbstractVector{T}
    βs::AbstractVector{S}
    αs::AbstractVector{S}
    pu::P
    layout::L                      # always ColumnLayout — a vector is orientation-free

    """Build initial scalar-Lanczos factorization state."""
    function LanczosFactorization(maxdim::Int, basis::OrthonormalBasis, x0::AbstractVector{T}, pu::P,
                                  layout::L=ColumnLayout()) where {T<:Number, P<:ProcessingUnit, L<:Layout}
        S = real(T)
        r = similar(x0)
        v_last = pu.zeros(T, size(x0))
        v_secondlast = pu.zeros(T, size(x0))

        αs = zeros(S, maxdim)
        βs = zeros(S, maxdim-1)

        return new{S,T,P,L}(0, basis, r, v_last, v_secondlast, βs, αs, pu, layout)
    end
end

# --- overlap / norm bookkeeping ---
"""Current scalar overlap `α_k`."""
@inline getoverlap(fact::LanczosFactorization) = fact.αs[fact.krylovdim]
"""
Current scalar norm-coupling value `β`. Returns `0` when only one vector is present
(`krylovdim < 2`), the situation that arises during a breakdown extraction — the
single-vector subspace has no coupling, so the Ritz residual estimate is `0`.
"""
@inline getnorm(fact::LanczosFactorization) =
    fact.krylovdim < 2 ? zero(eltype(fact.βs)) : fact.βs[fact.krylovdim-1]
"""Store scalar norm-coupling value `β` for the current iteration."""
@inline addnorm!(fact::LanczosFactorization, β::Real) = (fact.βs[fact.krylovdim-1] = β)

"""
Construct the projected matrix from stored Lanczos coefficients. On CPU this is a
`SymTridiagonal` (so `eigen` dispatches to the fast symmetric-tridiagonal solver);
on GPU it is materialized as a dense device matrix.
"""
@inline constructfactorizedmat(fact::LanczosFactorization) =
    isa(fact.pu, GPU) ?
        gpu_matrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) :
        SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])

"""Normalize the residual vector in place and return its norm."""
@inline calc_norm_krylovvec!(fact::LanczosFactorization) = (β = LinearAlgebra.norm(fact.r); fact.r ./= β; β)

"""Compute and store the scalar overlap `α_k` for the current iteration."""
@inline calcoverlap!(fact::LanczosFactorization) = (fact.αs[fact.krylovdim] = real(dot(fact.v_last, fact.r)))

"""
Advance scalar-Lanczos tridiagonalization by one iteration. Returns `false` on
**breakdown** (`β = ‖r‖` below the breakdown tolerance — an invariant subspace has
been reached), without dividing by the vanishing `β`; otherwise `true`.
"""
@inline function tridiagonalization!(fact::LanczosFactorization)
    β = norm(fact.r)
    β < _breakdown_tol(eltype(fact.r)) && return false
    fact.r ./= β
    fact.βs[fact.krylovdim] = β
    update!(fact, fact.r)
    return true
end
