
"""
    mutable struct LanczosFactorization{T,S<:Real} <: KrylovFactorization{T,S}

Structure to store a Lanczos factorization of a real symmetric or complex hermitian linear
map `A` of the form

```julia
A * V = V * B + r * b'
```

For a given Lanczos factorization `fact` of length `k = length(fact)`, the basis `V` is
obtained via [`basis(fact)`](@ref basis) and is an instance of [`OrthonormalBasis{T}`](@ref
Basis), with also `length(V) == k` and where `T` denotes the type of vector like objects
used in the problem. The Rayleigh quotient `B` is obtained as
[`rayleighquotient(fact)`](@ref) and is of type `SymTridiagonal{S<:Real}` with `size(B) ==
(k,k)`. The residual `r` is obtained as [`residual(fact)`](@ref) and is of type `T`. One can
also query [`normres(fact)`](@ref) to obtain `norm(r)`, the norm of the residual. The vector
`b` has no dedicated name but can be obtained via [`rayleighextension(fact)`](@ref). It
takes the default value ``e_k``, i.e. the unit vector of all zeros and a one in the last
entry, which is represented using [`SimpleBasisVector`](@ref).

A Lanczos factorization `fact` can be destructured as `V, B, r, nr, b = fact` with
`nr = norm(r)`.

`LanczosFactorization` is mutable because it can [`expand!`](@ref) or [`shrink!`](@ref).
See also [`LanczosIterator`](@ref) for an iterator that constructs a progressively expanding
Lanczos factorizations of a given linear map and a starting vector. See
[`ArnoldiFactorization`](@ref) and [`ArnoldiIterator`](@ref) for a Krylov factorization that
works for general (non-symmetric) linear maps.
"""
mutable struct LanczosFactorization{E,P} <: KrylovFactorization{E,P}
    krylovdim::Int # current Krylov dimension
    basis::Basis # basis of length k
    r:: AbstractVector{E}
    βs::AbstractVector{E}
    αs::AbstractVector{E}
    pu::P
    
    function LanczosFactorization(maxdim::Int, basis::Basis, x₀::AbstractVector{E}, pu::P) where {E, P<:ProcessingUnit}
        r  = similar(x₀)
        βs = Vector{E}(undef, maxdim-1)
        αs = Vector{E}(undef, maxdim)

        # println("Lanczos factorization test")
        # display(E)
        # display(typeof(r))
        # display(typeof(αs))
        # display(typeof(βs))
        return new{E,P}(0, basis, r, βs, αs, pu)
    end
end





# get factorization properties
@inline getoverlap(fact::LanczosFactorization) = fact.αs[fact.krylovdim]
@inline getnorm(fact::LanczosFactorization) = fact.βs[fact.krylovdim-1]

# add factorization properties 
@inline addnorm!(fact::LanczosFactorization, β::Real) = begin fact.βs[fact.krylovdim-1] = β; end


# other functions
@inline constructfactorizedmat(fact::LanczosFactorization) = begin isa(fact.pu, GPU) ? CuMatrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) : Matrix(SymTridiagonal(fact.αs[1:fact.krylovdim], fact.βs[1:fact.krylovdim-1])) end
@inline calc_norm_krylovvec!(fact::LanczosFactorization) = begin β = LinearAlgebra.norm(fact.r); fact.r ./= β; return β; end

@inline calcoverlap!(fact::LanczosFactorization) = begin fact.αs[fact.krylovdim] = dot(last(fact.basis), fact.r) end

@inline tridiagonalization!(fact::LanczosFactorization) = begin
    fact.krylovdim += 1

    β = norm(fact.r)
    fact.r ./= β
    fact.βs[fact.krylovdim-1] = β

    add!(fact.basis, fact.r)
end



