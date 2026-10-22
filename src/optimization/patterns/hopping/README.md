# `hopping` — the Anderson operator in any dimension

```
H = diag(ε) + t · Σ_{k=1}^{D} (T_k⁺ + T_k⁻)        periodic, rectangular, D-dimensional
y[i] = ε[i]·x[i] + t · Σ_k ( x[i + ê_k] + x[i - ê_k] )
```

On-site disorder plus one uniform nearest-neighbour amplitude. The neighbour is a site index ± a
lattice stride with one wrap test per axis, so this pattern stores **no index table** — 168 MB of
neighbour columns at 128³ that the shape tuple already implies, out of a `SparseMatrixCSC` that
is ~1.1 GB in the first place.

## Files

| file | what it holds |
|:--|:--|
| `HopStructure.jl` | the lattice (`shape`, `strides`, `diags`, `hopping`), its validation, `rescale`, and `_derive_hop` — recovering the lattice from a `MappingStructure` |
| `kernels/hop_kernel_common.jl` | `MAX_UNROLLED_DIM`, the per-line neighbour bases, the work decomposition — everything both kernels share |
| `kernels/hopping_packed_kernel.jl` | **packed**: `P = W/bs` adjacent sites per SIMD register |
| `kernels/hopping_rows_kernel.jl` | **rows**: one site's whole block per `SVector{bs}` register |
| `build_kernel.jl` | `(variant, structure, bs) → closure`, plus the matrix-free `build_hop_kernel` |
| `validate_kernel.jl` | `can_apply` / `why_not` / `auto_score` |

## The one idea both kernels share: slide axis 1

A `2D+1`-point stencil naively reads `2D+1` input streams per site. Both kernels instead carry
the axis-1 neighbours in registers:

```
next ──> cur ──> prev            each site is LOADED ONCE and used three times
```

so the stencil costs `2D-1` streams. The two kernels differ only in **what a register holds**,
and that single choice decides everything else:

| | `packed` | `rows` |
|:--|:--|:--|
| a register is | `P = W/bs` adjacent **sites**, one block index | one **site**, all `bs` blocks |
| axis-1 neighbours | straddle the register boundary → two `shufflevector`s | whole registers → plain rotation, **no shuffle** |
| element type | real only (SIMD.jl `Vec`) | anything, including `Complex` |
| arguments | dense `Matrix` (raw pointers) | any `AbstractMatrix` |
| block size | `bs ≤ W` and `bs | W` | any |
| x-extent | `P` must divide `L₁` (see below) | any |
| owns | small `bs`, where an `SVector{1}` is a scalar | everything the other refuses |

At `P = 1` the permutations degenerate to "all of `prev`" and "all of `next`" and `packed`
becomes `rows` exactly. That is why `_build_hop_packed` can always repair an illegal `W`: it
halves it until `P` divides `L₁`, and `P = 1` divides everything.

## Measured

Why slide at all — from the design sweep in `notebooks/benchmarks/matvec/hop` (Zen 4, D = 3,
L = 128, 16 threads, ms per vector). This is the *benchmark's* kernels, not these:

| bs | 1 | 2 | 4 | 8 | 16 |
|:--|--:|--:|--:|--:|--:|
| sliding | 0.100 | 0.073 | 0.094 | 0.139 | 0.170 |
| non-sliding stencil, same layout | 0.274 | 0.246 | 0.291 | 0.247 | 0.256 |

Sliding won **every one of the 90** `(L, bs, threads)` cells in that sweep, which is why both
kernels here are built on it.

And what the two kernels in this folder actually do (EPYC 9354, 16 threads, `Float64`, default
parameters, ms per vector, `packed` / `rows`):

| lattice | `bs = 1` | `bs = 4` | `bs = 8` |
|:--|--:|--:|--:|
| `128³` | 0.132 / 0.355 | 0.094 / 0.191 | 0.165 / 0.209 |
| `2048 × 1024` | 0.103 / 0.256 | 0.076 / 0.138 | 0.155 / 0.240 |
| `12⁶` | 0.780 / 1.719 | 0.298 / 0.700 | 0.499 / 0.617 |
| `2²¹` (1-D) | 0.062 / 0.159 | 0.057 / 0.166 | 0.141 / **0.129** |

`packed` leads by 1.8–2.7× at small `bs` and the two converge as the `SVector` grows into a
register of its own — in one dimension at `bs = 8` `rows` is already ahead. Tuning `W` moved
`128³, bs = 1` from 0.132 to 0.107; `tile = 64` moved `128³, bs = 2` from 0.086 to 0.078. Neither
default is a constant, which is why both are exposed.

## Dimension

`D` is a **type** parameter, never a field. The same arithmetic with the extents in a `Vector`
and a runtime loop over axes measured **3.7–5.7× slower**; that is the one non-negotiable finding
of the benchmark.

Above that, `MAX_UNROLLED_DIM = 6` splits how the *orthogonal* neighbours are summed:

- `D ≤ 6` — fully unrolled. Every one of the `2(D-1)` loads is straight-line code at an immediate
  offset, summed through a balanced tree.
- `D > 6` — a counted loop over the (still statically sized) tuple of line bases. The cut bounds
  **code**, not correctness: `D = 12` would put 22 live load streams plus their accumulator tree
  in one loop body and spill.

Either side of the cut, `D` is still a type parameter and all coordinate work is still hoisted
per x-line. Hand-written per-dimension kernels are deliberately **not** used: on the same
benchmark a generated per-`D` kernel tied hand-written fixed-`D` ones at a median ratio of 1.000
over `D = 1…6`, with outliers in both directions (at `D = 6, bs = 4` the generated one was
**0.89×**, i.e. faster). Six copies of the same loop would buy nothing measurable and would have
to be maintained in lockstep.

## Layout and the anisotropy rule

Sites are linearised **x-fastest**: `site = Σ_k x_k·s_k`, `s₁ = 1`, `s_{k+1} = s_k·L_k`. Blocks
are row-layout, `X[b, site]`, `bs × dim` — the same convention as every other kernel here.

Axis 1 is the slid one, so its neighbours are free while the other `2(D-1)` cost a load stream
each. **Put the longest axis first** on an anisotropic lattice.

## Two ways in

```julia
# 1. matrix-free — the one that matters at scale
hs = HopStructure(randn(prod(shape)), shape, -1.0)
f! = build_hop_kernel(rescale(hs, a, b), Val(bs))

# 2. from a matrix, through the ordinary entrypoint — `_derive_hop` recovers the lattice
polfed(H, x0, howmany, target; optimize_mapping = :hopping)
```

`_derive_hop` reconstructs the shape from row 1's neighbour offsets and then verifies **every**
row, in one `O(nnz)` pass. It is exact, not heuristic: anything that is not this operator returns
`nothing` and the caller falls through to `universal`.

## Extents below 3 are refused

On a periodic axis of length 2 the `+ê` and `-ê` neighbours are the same site, so this pattern
computes `2t` on that bond while the matrix a user writes down carries `t` — right by one
convention, silently wrong by the other. Length 1 makes a site its own neighbour. Both are
rejected at construction, where the message is legible.

## Parameters

`packed` takes `W` (pack width; `0` derives it from the hardware) and `tile`; `rows` takes
`tile` (x-lines per parallel chunk; `0` = one contiguous range per thread).

`W` is not "wider is better", and the direction flips with `bs`. Swept on `128³` at 16 threads
(ms per vector): at `bs = 1` the widest native pack wins — `W = 4`: 0.235, `W = 8`: 0.142,
**`W = 16`: 0.107** — and going past it to `W = 32` loses (0.187); at `bs = 8` the narrower pack
wins instead — **`W = 8`: 0.175** against `W = 16`: 0.216 — because `P = W/bs` is already small
and the wider shuffles stop paying for themselves. Sweep it with `benchmark_mapping`.
