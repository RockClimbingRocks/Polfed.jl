# Clenshaw

Fast, in-place evaluation of polynomial expansions of a linear operator via the
**Clenshaw recurrence**. Given expansion coefficients `c₀ … c_N` and a polynomial
family `φ` (Chebyshev, Legendre, …), this module evaluates

```
S(A) = Σₖ cₖ · φₖ(A)
```

where `A` is anything you can multiply by — a dense/sparse matrix, a matrix-free
operator, or (for the scalar path) a plain number. The recurrence is run
backwards (the numerically stable direction) and writes its result into a
caller-supplied buffer, so a configured expansion can be applied many times
without allocating.

This is a **POLFED-agnostic numerical primitive**: it constructs no POLFED
structs and can be unit-tested entirely against closed-form polynomial values.

---

## File layout

```
Clenshaw/
├── Clenshaw.jl                  «module root»
│       recurrence_weights(Val)      — per-family weights (concrete-typed, dispatched)
│       ClenshawExpansion{Fam,C,R,F} — the callable type + all constructors + callables
│       build_clenshaw_kernels       — turns a mapping! into recurrence/final-sum closures
│
└── Algorithm/                    the pure numerics (no type, no state)
    ├── recurrence_relations.jl   mapping_recurrence! (vec/mat + UniformScaling) · scalar_recurrence
    ├── final_sum.jl              mapping_finalsum!   (vec/mat + UniformScaling) · scalar_finalsum
    ├── clenshaw_algorithm.jl     the reverse-recurrence loop (matrix & scalar) + buffer rotation
    └── fused_clenshaw.jl         AxpbyClenshaw · RangeClenshaw — no per-step switch of thread pool
```

The split is **type/API vs. math**: `Clenshaw.jl` is *what the thing is* (the
`ClenshawExpansion` type, how you build it, how you call it); `Algorithm/` is
*how it works* (the individual recurrence steps). The `Algorithm/` files are
`include`d first so the constructors in `Clenshaw.jl` can wire them up.

---

## The type

```julia
struct ClenshawExpansion{Family, C, R<:Function, F<:Function}
    coefficients::C       # fast indexing closure over precomputed cₖ
    order::Int            # highest degree N
    D::Int                # operator dimension
    T::Type               # operator element type
    recurrence!::R        # the per-step kernel
    finalsum!::F          # the terminal-combination kernel
end
```

`Family` (a `Symbol` like `:Chebyshev`) is a **type parameter**, and the callbacks
and coefficient accessor are stored as **concrete type parameters** (`C`, `R`,
`F`). So every dispatch through them is specialized — *both* call paths are
type-stable, including scalar `S(x)` (the recurrence weights resolve statically
from `Family`, via `recurrence_weights(Val(Family))`). Recover the family with
`family(ce)`.

---

## Construction capabilities

Four constructors cover the ways you might supply the operator `A`:

| # | Signature | Use when … |
|---|-----------|------------|
| 1 | `ClenshawExpansion(type, coeffs, order, mapping!, D, T)` | you have an in-place action `mapping!(out, in)` (`out ← A·in`) and know `D`, `T` |
| 2 | `ClenshawExpansion(type, coeffs, order, recurrence!, finalsum!, D, T)` | you want to supply the **whole recurrence** yourself (e.g. a fused/optimized kernel) |
| 3 | `ClenshawExpansion(type, coeffs, order, mat::AbstractMatrix)` | you have a concrete dense or sparse matrix |
| 4 | `ClenshawExpansion(type, coeffs, order, op)` | you have any **matrix-free operator** supporting `mul!`, `size`, `eltype` (e.g. a `LinearOperators.LinearOperator`) — no dependency required |
| 5 | `ClenshawExpansion(type, coeffs, order)` | you only need **scalar evaluation** `S(x)` and have no operator (e.g. KPM density reconstruction) — the vector/matrix call path then errors with a clear message |

`coeffs` may be either:
- an `AbstractVector` `[c₀, c₁, …, c_N]`, or
- a `Function` `k -> cₖ`.

Either way the coefficients are **evaluated once at construction** into a concrete
vector, so applying the expansion does not re-evaluate them.

### Supported polynomial families

Defined by `recurrence_weights(::Val{family})` (and mirrored in the
`polynomial_properties` registry for introspection):

| `type` | family | recurrence |
|--------|--------|-----------|
| `:Chebyshev` | Chebyshev `Tₙ` | `Tₙ₊₁ = 2x·Tₙ − Tₙ₋₁` |
| `:Legendre`  | Legendre `Pₙ`  | `(n+1)Pₙ₊₁ = (2n+1)x·Pₙ − n·Pₙ₋₁` |
| `:Hermite`   | physicists' `Hₙ` | `Hₙ₊₁ = 2x·Hₙ − 2n·Hₙ₋₁` |
| `:Taylor`    | monomial `xⁿ`  | `S(x) = Σ cₖ xᵏ` |

An unknown `type` raises `ArgumentError`.

---

## Calling an expansion

A built `ClenshawExpansion` is callable in three modes.

### 1. Apply to a vector or block: `S(A)·X`

```julia
ce = ClenshawExpansion(:Chebyshev, coeffs, N, A)   # A is a matrix/operator
Y  = similar(v)
b  = [similar(v) for _ in 1:3]                      # 3-buffer workspace
ce(Y, v, b)                                          # Y ← S(A)·v
```

`X` may be a vector **or** a matrix (block of column vectors evaluated together).
The workspace `b` is a length-3 vector of buffers matching the shape/type of `Y`;
it is zeroed on entry, so it can be reused across calls. A workspace of the wrong
length raises `AssertionError`.

### 2. Form the full operator: `S(A)`

Pass a `UniformScaling` (`I`) as the seed and square matrix buffers:

```julia
Y = zeros(T, D, D)
b = [zeros(T, D, D) for _ in 1:3]
ce(Y, I, b)                                          # Y ← S(A) as a dense matrix
```

### 3. Evaluate the scalar polynomial: `S(x)`

```julia
ce(0.7)        # real
ce(0.5 + 0.2im) # complex — returns a number; workspace allocated internally
```

For scalars no caller buffer is needed (the workspace is three numbers). An
expansion built with the operator-free constructor (`ClenshawExpansion(type,
coeffs, order)`) supports **only** this mode; calling its vector/matrix path
raises an informative error.

---

## Supported element types

The vector/matrix path is generic over the buffer element type — tested with
`Float32`, `Float64`, `ComplexF32`, `ComplexF64`, and over dense & sparse
matrices and matrix-free operators. **Coefficients may be complex** (`c`/`c₀`
accept any `Number`), which supports complex spectral filters; the recurrence
*weights* (`α`, `β`) are always real.

---

## Design notes

- **One implementation per step.** The recurrence and final-sum each have a
  single mapping-backed implementation (plus a `UniformScaling` overload and a
  scalar overload). Earlier code carried a second, dead matrix path; it is gone.
- **`@inline` lives on the leaf math** (`mapping_recurrence!`, `mapping_finalsum!`,
  `scalar_*`), not on the kernel-factory.
- **Coefficients are precomputed** at construction (see above).
- **Backwards recurrence** (`k = N → 1`) is the numerically stable direction; the
  3-buffer workspace is rotated by an allocation-free `SVector` permutation after
  each step.

---

## Fused evaluators

`ClenshawExpansion` does each step as "operator call, then a threaded combine". When the operator
has its own threads (MKL, or a kernel opening its own `@threads` region), every step switches
thread pool twice, and on many threads that switch can cost more than the step. Two evaluators
avoid it; both are callable as `e(Y, X, b)` and take the same family/coefficients/order:

| evaluator | operator given as | one step |
| :--- | :--- | :--- |
| `AxpbyClenshaw(family, c, K, axpby!, add!)` | `axpby!(Y, X, α, β)`: `Y ← α·A·X + β·Y`; `add!(Y, c, X)`: `Y ← Y + c·X` | one accumulating call + one add, over the oldest buffer in place |
| `RangeClenshaw(family, c, K, range!, units; alignment, ntasks)` | serial `range!(Y, X, lo, hi)` over work units, disjoint writes | the whole recurrence in ONE `@threads :static` region, two spin barriers per step |

`RangeClenshaw` does the same arithmetic per element as `ClenshawExpansion`, so it is
bit-identical to it for any `ntasks`; `AxpbyClenshaw` agrees to rounding. The spin barrier
yields its core after a short spin — if the OS puts two tasks on one core, the one still working
only runs when the other yields. `RangeClenshaw` cannot run inside another `:static` region.

---

## Testing

Tests live in `test/Numerics/Clenshaw/` and mirror this layout. They assert
against **closed-form ground truth** rather than snapshots:

- scalar: `Tₙ(cos θ) = cos(nθ)`, plus explicit Legendre/Hermite/Taylor polynomials;
- matrix/operator: cross-checked against the matrix Chebyshev recurrence;
- all four constructors, all three call modes, multiple element types, edge cases
  (order 0, high order, zero coefficients), and error paths.

Run them with:

```julia
import Pkg; Pkg.test()                 # whole package
import Pkg; Pkg.test(coverage=true)    # with line-coverage instrumentation
```

The module currently has **100% line coverage**.
