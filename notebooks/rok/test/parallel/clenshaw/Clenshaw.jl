"Fast and precise evaluation of matrix/operator polynomial expansions via the Clenshaw recurrence."
module Clenshaw

using LinearAlgebra, SparseArrays, StaticArrays

"""Polynomial families supported by [`recurrence_weights`](@ref) / `ClenshawExpansion`."""
const SUPPORTED_FAMILIES = (:Chebyshev, :Legendre, :Hermite, :Taylor)

"""
    recurrence_weights(::Val{family}) -> (α, β, ϕ₀, ϕ₁)

Return the recurrence-weight functions `α(k)`, `β(k)` and normalization constants
`ϕ₀`, `ϕ₁` for a polynomial `family`.

Dispatching on `Val{family}` (rather than a `Dict` lookup) keeps the returned
closures **concretely typed**, so the scalar evaluation path — which resolves the
family at the type level via `ClenshawExpansion{family,…}` — is type-stable. An
unknown family raises `ArgumentError`.
"""
recurrence_weights(::Val{:Chebyshev}) = ((k) -> 2,                  (k) -> -1,           1, 1)
recurrence_weights(::Val{:Legendre})  = ((k) -> (2k + 1) / (k + 1), (k) -> -k / (k + 1), 1, 1)
recurrence_weights(::Val{:Hermite})   = ((k) -> 2,                  (k) -> -2 * k,       1, 2)
recurrence_weights(::Val{:Taylor})    = ((k) -> 1,                  (k) -> 0,            1, 1)
recurrence_weights(::Val{F}) where {F} =
    throw(ArgumentError("Unknown polynomial type :$F. Supported families: $(SUPPORTED_FAMILIES)"))

"""`is_supported_family(s::Symbol)` — whether `s` names a known polynomial family."""
is_supported_family(s::Symbol) = s in SUPPORTED_FAMILIES

"""
Registry of recurrence weights per family, `family => (α, β, ϕ₀, ϕ₁)`, derived
from [`recurrence_weights`](@ref). Kept for introspection; the type-stable scalar
path uses `recurrence_weights(Val(family))` directly.
"""
const polynomial_properties = Dict(f => recurrence_weights(Val(f)) for f in SUPPORTED_FAMILIES)

include("algorithm/recurrence_relations.jl")
include("algorithm/final_sum.jl")
include("algorithm/clenshaw_algorithm.jl")

"""
    build_clenshaw_kernels(mapping!, Val(family)) -> (recurrence!, finalsum!)
    build_clenshaw_kernels(mapping!, family::Symbol) -> (recurrence!, finalsum!)

Construct the recurrence and final-sum callbacks for a mapping-backed expansion.

The `family` selects the recurrence weights `α`, `β` and normalization constants
`ϕ₀`, `ϕ₁` via [`recurrence_weights`](@ref); the linear action `mapping!` and those
weights are captured in two closures matching the signatures expected by
[`clenshaw_algorithm!`](@ref). Passing the family as `Val{family}` captures `α`/`β`
with concrete types, so the matrix hot loop dispatches them statically.

# Arguments
- `mapping!::Function`: in-place linear action `mapping!(out, in)` writing `A·in` into `out`.
- `family`: polynomial family, as a `Symbol` or `Val{family}`; one of [`SUPPORTED_FAMILIES`](@ref).

# Returns
- `(recurrence!, finalsum!)`, the step and terminal callbacks.
"""
function build_clenshaw_kernels(mapping!::Function, ::Val{Fam}) where {Fam}
    α, β, ϕ₀, ϕ₁ = recurrence_weights(Val(Fam))

    recurrence! = (b1::AbstractVecOrMat, b2::AbstractVecOrMat, b3::AbstractVecOrMat, c::Number, k::Int, X) ->
        mapping_recurrence!(mapping!, b1, b2, b3, c, α(k), β(k + 1), X)

    finalsum! = (b2::AbstractVecOrMat, b3::AbstractVecOrMat, c₀::Number, Y::AbstractVecOrMat, X) ->
        mapping_finalsum!(mapping!, b2, b3, c₀, β(1), ϕ₀, ϕ₁, Y, X)

    return recurrence!, finalsum!
end

build_clenshaw_kernels(mapping!::Function, family::Symbol) = build_clenshaw_kernels(mapping!, Val(family))

"""
    _precompute_coefficients(coefficients, order) -> Function

Materialize the expansion coefficients `c₀ … c_order` into a concrete vector at
construction time and return a fast indexing closure `k -> cᵥ[k+1]`.

Accepts either a `Function` (evaluated at `0:order`) or an `AbstractVector`
(indexed `1:order+1`). Precomputing removes `order + 1` coefficient evaluations
from every subsequent application of the expansion.
"""
function _precompute_coefficients(coefficients::Function, order::Int)
    cvec = [coefficients(k) for k in 0:order]
    return k::Int -> @inbounds cvec[k + 1]
end

"""`AbstractVector` overload of [`_precompute_coefficients`](@ref); checks length, then copies `c₀ … c_order`."""
function _precompute_coefficients(coefficients::AbstractVector{<:Number}, order::Int)
    length(coefficients) ≥ order + 1 ||
        throw(ArgumentError("coefficients has length $(length(coefficients)) but order $order needs at least $(order + 1) entries."))
    cvec = collect(coefficients[1:order+1])
    return k::Int -> @inbounds cvec[k + 1]
end

"""
    ClenshawExpansion{Family,C,R,F}

Callable object that evaluates a polynomial expansion via the Clenshaw recurrence.
Once built, it applies `S(A)` to a vector/matrix, forms the full `S(A)` (with a
`UniformScaling` seed), or evaluates the scalar polynomial `S(x)`.

The polynomial `Family` is a **type parameter** (a `Symbol`), so the scalar path
resolves the recurrence weights statically — `S(x)` is type-stable. The
recurrence/final-sum callbacks and the coefficient accessor are likewise stored as
concrete type parameters (`C`, `R`, `F`), so no dispatch in either hot loop is
dynamic.

# Fields
- `coefficients::C`: fast indexing closure over the precomputed coefficients.
- `order::Int`: highest polynomial degree.
- `D::Int`: problem dimension (rows of the operator).
- `T::Type`: element type of the operator.
- `recurrence!::R`, `finalsum!::F`: the Clenshaw step and terminal callbacks.

The family symbol is recovered with [`family`](@ref).

# Construction
Build from any of:
- a mapping callback `mapping!(out, in)` plus `(D, T)`,
- explicit `recurrence!` / `finalsum!` callbacks plus `(D, T)` — the "whole
  recurrence relation" path,
- a concrete `AbstractMatrix`,
- any object `op` supporting `mul!(out, op, in)`, `size`, and `eltype`
  (e.g. a `LinearOperators.LinearOperator`),
- *coefficients and order only* — a **scalar-only** expansion with no operator;
  the vector/matrix call path is then unavailable (see the no-operator method).

See the constructor methods below.
"""
struct ClenshawExpansion{Family, C, R<:Function, F<:Function}
    coefficients::C
    order::Int
    D::Int
    T::Type
    recurrence!::R
    finalsum!::F
end

"""    family(c::ClenshawExpansion) -> Symbol

The polynomial family symbol (e.g. `:Chebyshev`) carried in the type parameter."""
family(::ClenshawExpansion{Family}) where {Family} = Family

"""
    ClenshawExpansion(type, coefficients, order, mapping!, D, T)

Build from a mapping callback `mapping!(out, in)` (`out ← A·in`) and the problem
metadata `D` (dimension) and `T` (element type). The recurrence/final-sum
kernels are derived from `type` via [`build_clenshaw_kernels`](@ref).
"""
function ClenshawExpansion(
    type::Symbol,
    coefficients::Union{Function,AbstractVector{<:Number}},
    order::Int,
    mapping!::Function,
    D::Int,
    T::Type,
)
    is_supported_family(type) ||
        throw(ArgumentError("Unknown polynomial type :$type. Supported families: $(SUPPORTED_FAMILIES)"))
    coeff = _precompute_coefficients(coefficients, order)
    recurrence!, finalsum! = build_clenshaw_kernels(mapping!, Val(type))
    return ClenshawExpansion{type, typeof(coeff), typeof(recurrence!), typeof(finalsum!)}(coeff, order, D, T, recurrence!, finalsum!)
end

"""
    ClenshawExpansion(type, coefficients, order, recurrence!, finalsum!, D, T)

Build from explicit `recurrence!` and `finalsum!` callbacks. Use this when the
recurrence is supplied directly (e.g. a fused/optimized kernel) rather than
derived from a generic `mapping!`.
"""
function ClenshawExpansion(
    type::Symbol,
    coefficients::Union{Function,AbstractVector{<:Number}},
    order::Int,
    recurrence!::Function,
    finalsum!::Function,
    D::Int,
    T::Type,
)
    is_supported_family(type) ||
        throw(ArgumentError("Unknown polynomial type :$type. Supported families: $(SUPPORTED_FAMILIES)"))
    coeff = _precompute_coefficients(coefficients, order)
    return ClenshawExpansion{type, typeof(coeff), typeof(recurrence!), typeof(finalsum!)}(coeff, order, D, T, recurrence!, finalsum!)
end

"""
    ClenshawExpansion(type, coefficients, order, mat::AbstractMatrix)

Build from a concrete matrix, with the linear action `out ← mat·in` via `mul!`.
`D` and `T` are taken from `mat`.
"""
function ClenshawExpansion(
    type::Symbol,
    coefficients::Union{Function,AbstractVector{<:Number}},
    order::Int,
    mat::AbstractMatrix{<:Number},
)
    mapping!(Ỹ, Y) = mul!(Ỹ, mat, Y)
    return ClenshawExpansion(type, coefficients, order, mapping!, size(mat, 1), eltype(mat))
end

"""
    ClenshawExpansion(type, coefficients, order, op)

Build from any linear operator `op` that supports `mul!(out, op, in)`, `size`,
and `eltype`. This is the duck-typed path for `LinearOperators.LinearOperator`
and similar matrix-free operators; no dependency on `LinearOperators.jl` is
required.
"""
function ClenshawExpansion(
    type::Symbol,
    coefficients::Union{Function,AbstractVector{<:Number}},
    order::Int,
    op,
)
    mapping!(Ỹ, Y) = mul!(Ỹ, op, Y)
    return ClenshawExpansion(type, coefficients, order, mapping!, size(op, 1), eltype(op))
end

"""
    _scalar_only_kernel(args...)

Placeholder recurrence/final-sum kernel installed on a **scalar-only**
[`ClenshawExpansion`](@ref) (one built from coefficients and order alone, with no
operator). Invoking the vector/matrix call path on such an expansion reaches this
kernel and raises an informative error.
"""
_scalar_only_kernel(args...) = error(
    "This ClenshawExpansion was built for scalar evaluation only (no operator was provided). " *
    "Rebuild it with a mapping!, matrix, or operator to use the vector/matrix call path."
)

"""
    ClenshawExpansion(type, coefficients, order)

Build a **scalar-only** expansion: one that evaluates the polynomial `S(x)` at a
number but carries no operator. No `mapping!`/matrix is needed (and none is
fabricated). `D` is set to `0` and `T` to the coefficient element type; the
`recurrence!`/`finalsum!` fields are set to [`_scalar_only_kernel`](@ref), so the
vector/matrix call path raises a clear error if used.

Useful wherever only scalar evaluation is required — e.g. reconstructing a KPM
density `ρ(x) = Σₖ cₖ Tₖ(x) / (π√(1-x²))`.
"""
function ClenshawExpansion(
    type::Symbol,
    coefficients::Union{Function,AbstractVector{<:Number}},
    order::Int,
)
    is_supported_family(type) ||
        throw(ArgumentError("Unknown polynomial type :$type. Supported families: $(SUPPORTED_FAMILIES)"))
    coeff = _precompute_coefficients(coefficients, order)
    T = coefficients isa AbstractVector ? eltype(coefficients) : typeof(coefficients(0))
    return ClenshawExpansion{type, typeof(coeff), typeof(_scalar_only_kernel), typeof(_scalar_only_kernel)}(
        coeff, order, 0, T, _scalar_only_kernel, _scalar_only_kernel,
    )
end

"""
    (clenshaw::ClenshawExpansion)(Y, X, b) -> Y

Apply the expansion to the vector/matrix `X`, writing the result into `Y`.

`X` may be an `AbstractVecOrMat` (compute `S(A)·X`) or a `UniformScaling` such as
`I` (form the full `S(A)`). `b` is a length-3 workspace of buffers matching the
type/shape of `Y`; it is zeroed on entry, so it can be reused across calls.
"""
function (clenshaw::ClenshawExpansion)(
    Y::AbstractVecOrMat{<:Number},
    X::Union{AbstractVecOrMat{<:Number},UniformScaling},
    b::AbstractVector{<:AbstractVecOrMat{<:Number}},
)
    @assert length(b) == 3 "workspace `b` must have length 3, got $(length(b))"
    b[1] .= 0
    b[2] .= 0
    b[3] .= 0

    return clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, clenshaw.recurrence!, clenshaw.finalsum!, b, X, Y)
end

"""
    (clenshaw::ClenshawExpansion)(x::Number) -> Number

Evaluate the scalar polynomial represented by `clenshaw` at `x` (real or complex).
The recurrence weights are resolved statically from the `Family` type parameter,
so this path is type-stable. Workspace is allocated internally — for scalars this
is three numbers, so no caller-managed buffer is needed.
"""
function (clenshaw::ClenshawExpansion{Family})(x::Number) where {Family}
    α, β, ϕ₀, ϕ₁ = recurrence_weights(Val(Family))
    b = zeros(typeof(x), 3)
    return clenshaw_algorithm!(clenshaw.coefficients, clenshaw.order, α, β, ϕ₀, ϕ₁, b, x)
end

export ClenshawExpansion

end # module Clenshaw
