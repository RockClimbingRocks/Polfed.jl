# [Optimized kernels](@id optimized_kernels)

Lever 1 from [The bottleneck](@ref going_faster): make ``H\cdot v`` cheaper. POLFED is
memory-bound, so this is about moving fewer bytes, not doing fewer FLOPs.

## The automatic path

One keyword:

```julia
mc = MappingConfig(optimize_mapping = true)
result = polfed(H, x0, howmany, target; mc = mc)
display_report(result)
```

With matrix input, POLFED inspects the structure of `H` and builds specialized row kernels.
The idea:

- split the diagonal and offdiagonal contributions,
- **group repeated offdiagonal values** — store each distinct value once and keep only the
  connectivity,
- avoid redundant loads and stores in the repeated application,
- reuse the same kernels inside the Clenshaw steps of the transform.

Because the transform applies the operator ``K`` times per Lanczos iteration, even a modest
per-call saving compounds into a large end-to-end speedup.

`optimize_mapping` accepts more than `true`: a kernel name, or a `(kernel = …, …)` named
tuple carrying that kernel's own tuning parameters, when you want to pin the choice rather
than let POLFED pick. Which kernels exist and what they take is an implementation detail that
changes with the hardware POLFED is tuned for — `true` is the supported entry point. See
[`MappingConfig`](@ref UniPolfed.MappingConfig).

### When it helps

This path pays off exactly when the Hamiltonian has **few distinct offdiagonal values**. A
spin chain in the ``S^z`` basis typically has one. In that regime there is no reason to look
up, load and multiply a value that is the same everywhere — POLFED can hoist it out and
touch only indices.

The Quantum Sun model benefits from the structural split. The XXZ model, with a single
constant offdiagonal value, benefits far more.

## [A worked example: XXZ](@id optimized_kernels_xxz)

The XXZ spin chain is

```math
H = \sum_{i=1}^{L}\left(S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + \Delta S_i^z S_{i+1}^z\right)
  = \sum_{i=1}^{L}\left(\tfrac{1}{2}(S_i^{+}S_{i+1}^{-}+S_i^{-}S_{i+1}^{+}) + \Delta S_i^z S_{i+1}^z\right),
```

with periodic boundary conditions (site ``L+1`` identified with site ``1``). What there is
to exploit:

- sparsity,
- regular, coalescible memory access in the mapping loop,
- a **constant** offdiagonal value in this representation,
- known offdiagonal connectivity.

### Extract a compact representation

```julia
using UniPolfed.Models: xxz_hamiltonian
using SparseArrays

construct_XXZ_matrix(L::Int, Delta::Real, Lup::Int) =
    xxz_hamiltonian(L, Lup, 1.0, Delta, 0.0;
        boundary=:periodic, field=0.0, use_sparse=true)

function get_diags_and_offdiagonals_single_value(
    Delta::Real, L::Int, Lup::Int; tol=1e-13, round_digits=14,
)
    mat = construct_XXZ_matrix(L, Delta, Lup)
    dim = size(mat, 1)
    diagonals = [round(mat[i, i]; digits=round_digits) for i in 1:dim]
    flat = Int[]
    starts = Int[]
    idx = 1
    offdiag_val::Union{Nothing, Float64} = nothing

    for i in 1:dim
        push!(starts, idx)
        for col in nzrange(mat, i)
            j = rowvals(mat)[col]
            i == j && continue
            v = mat[i, j]
            abs(v) < tol && continue

            v_rounded = round(v; digits=round_digits)
            if offdiag_val === nothing
                offdiag_val = v_rounded
            elseif abs(v_rounded - offdiag_val) > tol
                error("Matrix has multiple distinct off-diagonal values (found $v_rounded and $offdiag_val).")
            end
            push!(flat, j)
        end
        idx += length(flat) - starts[end] + 1
    end

    offdiag_val === nothing && error("No off-diagonal elements found above tolerance.")
    return diagonals, offdiag_val, flat, starts
end
```

This returns one diagonal value per row (`diagonals`), the single shared offdiagonal value
(`offdiag_val`), the flattened offdiagonal column indices (`flat`), and row-start pointers
into that flat array (`starts`).

### Build the mapping

```julia
function mapvec_with_xxz!(
    diags::Vector{Float64},
    offdiag_val::Float64,
    offdiags_flatten::Vector{Int},
    start_indices::Vector{Int},
)
    return (Y, X) -> begin
        for i in eachindex(start_indices)
            start = start_indices[i]
            @inbounds stop = (i == length(start_indices)) ? length(offdiags_flatten) : start_indices[i+1] - 1
            sum_val = 0.0
            for j in start:stop
                @inbounds sum_val += X[offdiags_flatten[j]]
            end
            @inbounds Y[i] = muladd(diags[i], X[i], offdiag_val * sum_val)
        end
    end
end
```

The inner loop reads only indices and accumulates; the shared value is multiplied once per
row. No value lookups, no generic sparse traversal.

### Benchmark the kernel before the solver

```julia
using BenchmarkTools
using LinearAlgebra

L, Lup, Delta = 18, 9, 0.55
mat = construct_XXZ_matrix(L, Delta, Lup)
diags, offdiag_val, flat, starts = get_diags_and_offdiagonals_single_value(Delta, L, Lup)
f_custom! = mapvec_with_xxz!(diags, offdiag_val, flat, starts)

X = rand(size(mat, 1)); Y = similar(X)
@btime mul!($Y, $mat, $X)
@btime $f_custom!($Y, $X)
```

Isolating the kernel keeps the measurement fast and repeatable, and avoids attributing
allocation noise from a full `polfed` run to the mapping.

```@raw html
<table>
  <thead>
    <tr>
      <th style="text-align:center;">L</th>
      <th style="text-align:center;">generic <code>mul!</code></th>
      <th style="text-align:center;">custom mapping</th>
      <th style="text-align:center;">speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:right;">18</td><td style="text-align:right;">856.653 μs</td><td style="text-align:right;">579.533 μs</td><td style="text-align:right;">1.48</td></tr>
    <tr><td style="text-align:right;">20</td><td style="text-align:right;">5.402 ms</td><td style="text-align:right;">2.340 ms</td><td style="text-align:right;">2.31</td></tr>
    <tr><td style="text-align:right;">22</td><td style="text-align:right;">25.727 ms</td><td style="text-align:right;">10.202 ms</td><td style="text-align:right;">2.52</td></tr>
    <tr><td style="text-align:right;">24</td><td style="text-align:right;">109.943 ms</td><td style="text-align:right;">41.825 ms</td><td style="text-align:right;">2.63</td></tr>
  </tbody>
</table>
```

The advantage grows with system size: at ``L=18`` it is about ``1.5\times``, and it settles
around ``2.5\text{–}3\times`` for larger chains, where memory traffic dominates completely.

### Use it in POLFED

```julia
x0 = normalize!(rand(size(mat, 1)))
howmany, target = 120, 0.0

result_mul    = polfed(mat, x0, howmany, target)        # generic path
result_custom = polfed(f_custom!, x0, howmany, target)  # custom kernel

display_report(result_mul)
display_report(result_custom)
```

The end-to-end speedup should track the kernel benchmark, because the mapping dominates the
total work.

### Then remove the rescaling overhead

The transform runs on ``\tilde H = (H-b)/a``. If you only supply `f!`, POLFED rescales at
runtime — extra vector passes in the hottest loop. Supply the rescaled action directly:

```julia
Emin, Emax = UniPolfed.Numerics.Lanczos.lanczos_extrema(f_custom!, x0; maxdim=1000)
spread = (Emax - Emin) / 2
center = (Emax + Emin) / 2

diags_rescaled       = @. (diags - center) / spread
offdiag_val_rescaled = offdiag_val / spread
f_rescaled!          = mapvec_with_xxz!(diags_rescaled, offdiag_val_rescaled, flat, starts)

mc = MappingConfig(f!_rescaled = f_rescaled!, Emin = Emin, Emax = Emax)
result = polfed(f_custom!, x0, howmany, target; mc = mc)
display_report(result)
```

Note the rescaling is folded into the *data*, not applied as extra passes — the kernel is
unchanged. This is precisely what `optimize_mapping=true` does for you automatically, which
is why the automatic path reaches nearly the same speed on this model with none of the code
above.

Even without a structured kernel, the cheap version of this trick helps: build a rescaled
copy of the matrix once and use `mul!(Y, mat_rescaled, X)` as `f!_rescaled`.

## Automatic or custom?

| | `optimize_mapping=true` | your own kernel |
| :--- | :--- | :--- |
| Effort | one keyword | write and validate a kernel |
| Needs | matrix input, few distinct offdiagonal values | anything you can express |
| Speed on structured models | close to hand-written | the ceiling |
| Works with operator input | no | yes |

Start automatic. Write your own when the automatic path cannot apply — an operator with no
assembled matrix — or when profiling shows the kernel is still the wall.

## Next

- [Parallelization](@ref parallelization) — run these kernels on many cores.
- [GPU](@ref going_faster_gpu) — the same ideas on device.
- [Bring your own operator](@ref operator_interface) — the interface details.
