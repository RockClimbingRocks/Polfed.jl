# [Optimization of the XXZ Model](@id optimization_xxz_model)

This section introduces the XXZ spin chain and explains how to exploit its
structure to build faster mapping kernels for POLFED. For the public model
constructor itself, see [XXZ](@ref xxz_model).

The Hamiltonian is

```math
H = \sum_{i=1}^{L}\left(S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + \Delta S_i^z S_{i+1}^z\right)
  = \sum_{i=1}^{L}\left(\frac{1}{2}(S_i^{+}S_{i+1}^{-}+S_i^{-}S_{i+1}^{+}) + \Delta S_i^z S_{i+1}^z\right).
```

Here ``L`` is system size, ``S_i^x``, ``S_i^y``, and ``S_i^z`` are spin-1/2
operators at site ``i``, and the raising/lowering operators are

```math
S_i^\pm = S_i^x \pm i S_i^y.
```

The nearest-neighbor index is taken with periodic boundary conditions, so site
`L+1` is identified with site `1`.

## What Can We Exploit

- Hamiltonian sparsity.
- Regular/coalesced memory access patterns in mapping loops.
- Constant offdiagonal value in this XXZ representation.
- Known offdiagonal connectivity structure.

These properties let us replace a generic, well optimized sparse `mul!`
mapping with a structure-aware custom mapping.

## Matrix Construction for XXZ

In the public API, the XXZ Hamiltonian is constructed through
[`xxz_hamiltonian`](@ref Polfed.Models.xxz_hamiltonian) in `Polfed.Models`.
For the optimization workflow below we use a small wrapper that fixes
`J = 1.0`, `W = 0.0`, `field = 0.0`, and periodic boundary conditions.

```julia
using Polfed.Models: xxz_hamiltonian

construct_XXZ_matrix(L::Int, Delta::Real, Lup::Int) =
    xxz_hamiltonian(
        L,
        Lup,
        1.0,
        Delta,
        0.0;
        boundary=:periodic,
        field=0.0,
        use_sparse=true,
    )
```

## Construct Diagonal and Offdiagonal Data

The next function builds the XXZ matrix from `Delta`, `L`, and `Lup`, then
preprocesses it into compact arrays used by the custom mapper:
- `diagonals`: one diagonal value per row,
- `offdiag_val`: the common offdiagonal value,
- `flat`: flattened offdiagonal column indices,
- `starts`: row-start pointers into `flat`.

```julia
function get_diags_and_offdiagonals_single_value(
    Delta::Real,
    L::Int,
    Lup::Int;
    tol=1e-13,
    round_digits=14,
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

## Build a Custom Vector Mapping

This function returns a mapping closure `f_custom!(Y, X)` that computes
``H X \rightarrow Y`` using the compact XXZ representation:

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

In the next advanced pages, this mapping is integrated directly into
[`polfed`](@ref Polfed.polfed) workflows.
*

