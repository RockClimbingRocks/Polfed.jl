# [GPU](@id going_faster_gpu)

POLFED spends nearly all of its time in the spectral transformation — repeated sparse
matrix-vector or matrix-matrix products inside the polynomial filter. That workload is
data-parallel, memory-bound and regular, which is what GPUs are built for.

The interface barely changes: put `H` and `x0` on the device and call the same entrypoint.
Every entrypoint ([`polfed`](@ref Polfed.polfed), [`patched_polfed`](@ref Polfed.patched_polfed),
[`full_polfed`](@ref Polfed.full_polfed), [`double_polfed`](@ref Polfed.double_polfed)) runs on
the GPU, with scalar or block Lanczos, in either block layout, for real and complex `H`.

## Running on the GPU

```julia
using Polfed, CUDA, LinearAlgebra
using Polfed.Models: qsun_hamiltonian

H  = qsun_hamiltonian(12, 2, 1.0, 0.5)              # SparseMatrixCSC on the host
x0 = normalize!(rand(size(H, 1)))

H_gpu  = CUDA.CUSPARSE.CuSparseMatrixCSR(H)
result = polfed(H_gpu, CuVector(x0), 80, 0.0)
display_report(result)
```

Loading `CUDA` activates the package extension, which stays dormant otherwise. `H` and `x0`
must be on the same device — a host `x0` with a device `H` (or the reverse) is an
`ArgumentError`. CSC and COO device matrices are converted to CSR once; a dense `CuMatrix`
works too, but for any realistic many-body Hamiltonian use the sparse CSR form.

!!! note "CUDA.jl version"
    Use CUDA.jl 6 or newer on Grace Hopper (GH200) nodes. CUDA.jl 5.8 crashes intermittently
    there when reading a scalar back from the device (`dot`, `norm`), independently of POLFED.

## What runs where

Every length-`D` array — the Krylov basis, the filter's Clenshaw vectors, the KPM probes —
lives on the device. Everything projected — the block-tridiagonal matrix, its
eigendecomposition, the Ritz vectors and the coupling blocks — is small and lives on the host.
Each Lanczos step moves one `bs × bs` block across, so the device does the work that scales
with `D` and nothing else.

The returned eigenvalues and residuals are host vectors. The eigenvectors stay on the device:

```julia
vals = result.vals                       # host Vector
v1   = Array(eigenvector(result, 1))     # one eigenvector, on the host
V    = Array(copy(result.vecs))          # all of them
```

`copy` first: a row-layout result's `vecs` is a lazy `transpose` of a device view, which
`Array` cannot convert in one step.

## Block layout and block size

The block size is the smaller dimension of `x0`. A vector `x0` runs scalar Lanczos; a one-column
(or one-row) matrix runs block Lanczos with a block of one. The **layout** is how a block is
stored and multiplied:

- **column** — `x0` is `D × bs`, each step computes `H·X` (a column-major sparse × dense product);
- **row** — `x0` is `bs × D`, each step computes `X·Hᵀ` (a row-major product over the same CSR).

The matrix entrypoints accept `x0` in either shape and pick the layout themselves: on the GPU,
column below block size `Defaults.polfed.gpu_row_layout_from = 8` and row from it on (on the CPU
always row). Override it with `MappingConfig(layout = …)`:

```julia
x0_block = CUDA.randn(Float64, size(H, 1), 16)     # D × 16 — or 16 × D, either works
result = polfed(H_gpu, x0_block, 80, 0.0)           # auto: row layout
result = polfed(H_gpu, x0_block, 80, 0.0; mc = MappingConfig(layout = ColumnLayout()))
```

### Cost of one product, per vector

CUSPARSE on a GH200, XXZ chain ``L = 22`` (``D = 705\,432``, 12.5 nonzeros per row), `Float64`,
microseconds per vector — lower is better:

| block size | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| column (`H·X`) | 48 | 92 | 73 | 66 | 59 | 59 | 59 |
| row (`X·Hᵀ`) | 48 | 180 | 97 | 52 | 25 | 23 | 15 |

A row block amortizes reading `H` over all its vectors, so from block size 8 on it is the
cheapest per vector — 3.3× cheaper than a single vector at block size 64 (3.0× for `Float32`,
2.1× for `ComplexF64`, 3.2–3.7× at ``L = 24`` and on the QSun model). A column block never beats
a single vector.

### End to end

A block is cheaper per vector, but block Lanczos needs **more vectors** to converge the same
eigenpairs: the Krylov dimension grows roughly like ``\texttt{bs}^{0.6}`` while
``\texttt{howmany}/\texttt{bs}`` is small (see [`expected_krylov_dim`](@ref Polfed.expected_krylov_dim)).
The same XXZ chain, `howmany = 64` at the maximum of the density of states (filter order
``K \approx 91\,400``), wall time on a GH200:

| block size | column | row | Krylov vectors (column / row) |
|:---|---:|---:|---:|
| 1 | **889 s** | 927 s | 148 / 153 |
| 4 | 2224 s | 2512 s | 304 / 284 |
| 8 | 2481 s | 1710 s | 384 / 344 |
| 16 | 2916 s | 1591 s | 496 / 576 |

Wider column blocks only get slower: 5203 s at block size 32 and 7751 s at 64 (896 and 1344
Krylov vectors). Row blocks wider than 16 were not run end to end. Every run reached residuals
below ``10^{-13}``, and every run returned all 64 pairs except the column run at block size 32,
which returned 62. The block-size-1 times were measured before single vectors moved to a faster
CUSPARSE product (about 10% per product at this size), so scalar runs are now a little faster
than shown.

The layout rule picks the faster layout at every block size measured. The block size is yours to
choose (it is the shape of `x0`):

- **Start with a vector `x0`.** Scalar Lanczos was the fastest here; the best block (row, 16)
  took 1.8× as long.
- **A block pays off only for many eigenpairs per run.** Extrapolating the Krylov growth of
  [`expected_krylov_dim`](@ref Polfed.expected_krylov_dim), a row block of 16 should overtake a
  vector at a few hundred eigenpairs (about 230–360), a block of 8 only near 900. This is an
  estimate, not a measurement.
- **If you use a block, make it at least 8 wide**, so it runs in the row layout (the default
  from 8 on). A column block was slower than a vector at every size measured.

## Operator entrypoint on the GPU

[`polfed`](@ref Polfed.polfed) also takes a function `f!(Y, X)` that applies `H` to a column
block on the device:

```julia
f! = (Y, X) -> mul!(Y, H_gpu, X)
result = polfed(f!, CuVector(x0), 80, 0.0)
```

The operator path is column-only and builds the rescaled operator ``(H - b)/a`` around your
`f!` on every application. To do the rescaling once, in the data, pass `MappingConfig(f!_rescaled
= …)` together with the bounds it was built from (the [rescaling advice](@ref operator_interface)
applies unchanged). The matrix entrypoints do all of this themselves, which is why they are the
faster default.

## Custom Clenshaw kernels on the GPU

For model-specific tuning, supply the Clenshaw steps yourself. They replace the products with
the **rescaled** operator ``\tilde H = (H - b)/a``, so build it from bounds you also hand to
POLFED. They receive blocks in the run's layout; the templates below are written for column
blocks, so use them with a vector `x0` or with `layout = ColumnLayout()`:

```julia
Emin, Emax = lanczos_extrema(H_gpu)
a, b = (Emax - Emin) / 2, (Emax + Emin) / 2
H̃_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR((H - b * I) / a)

function make_gpu_clenshaw_kernels(H̃_gpu)
    crr = (b1, b2, b3, c, X) -> begin
        tmp = similar(X)
        mul!(tmp, H̃_gpu, b2)
        @. b1 = c * X + 2 * tmp - b3
        return nothing
    end

    cfs = (b1, b2, c, Y, X) -> begin
        tmp = similar(X)
        mul!(tmp, H̃_gpu, b1)
        @. Y = c * X + tmp - b2
        return nothing
    end

    return crr, cfs
end

crr_gpu!, cfs_gpu! = make_gpu_clenshaw_kernels(H̃_gpu)
mc = MappingConfig(clenshaw_recurrence = crr_gpu!, clenshaw_finalsum = cfs_gpu!,
                   Emin = Emin, Emax = Emax)

result = polfed(H_gpu, CuVector(x0), 80, 0.0; mc = mc)
```

These are templates, not production kernels — each allocates a temporary per call, which is
the first thing to remove in a fused model-specific implementation.

## Limits

- `optimize_mapping` (the value-grouped kernels of [Optimized kernels](@ref optimized_kernels))
  is CPU-only; asking for it with a device matrix is an `ArgumentError`. CUSPARSE plays that
  role on the GPU.
- The density of states draws its random probes from the device generator, so the estimated
  filter order differs from a CPU run of the same problem by the stochastic error (typically a
  few percent). The eigenvalues agree to the requested tolerance.

## Reading the output

- **Mapping → processing unit** should say GPU.
- **Timings** — CPU time is deliberately **suppressed** on GPU runs, where it measures host
  bookkeeping rather than device work and would mislead. Compare wall time.

Element type matters less on a GH200 than on consumer cards, which have little double-precision
throughput: there `Float32` saves 25–30% per product (it halves the vector and value traffic,
not the sparse index traffic). Set the tolerances accordingly — `eigentol = 1e-9` is not reachable in
single precision.

## Next

- [Optimized kernels](@ref optimized_kernels) — the CPU-side equivalent.
- [Guidelines](@ref guidelines) — settings by problem size.
