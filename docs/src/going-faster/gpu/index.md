# [GPU](@id going_faster_gpu)

POLFED spends nearly all of its time in the spectral transformation — repeated
matrix-vector or matrix-matrix products inside the polynomial filter. That workload is
massively data-parallel, memory-bound, and regular in its access pattern, which is a precise
description of what GPUs are built for.

The interface barely changes: move the arrays to the device and call
[`polfed`](@ref Polfed.polfed) exactly as before.

## Baseline on CPU

```julia
using Polfed
using Polfed.Models: qsun_hamiltonian
using LinearAlgebra

H_cpu  = qsun_hamiltonian(12, 2, 1.0, 0.5; use_sparse=true)
x0_cpu = normalize!(rand(size(H_cpu, 1)))

howmany, target = 80, 0.0
result_cpu = polfed(H_cpu, x0_cpu, howmany, target)
display_report(result_cpu)
```

## The same run on GPU

```julia
using CUDA

H_gpu  = CUDA.CUSPARSE.CuSparseMatrixCSR(H_cpu)
x0_gpu = CuVector(x0_cpu)

result_gpu = polfed(H_gpu, x0_gpu, howmany, target)
display_report(result_gpu)
```

That is the entire change: array type. Loading `CUDA` activates the package extension, which
stays dormant otherwise. Compare `result_cpu.report` with `result_gpu.report` — the
matrix-multiplication counts should match, and only the timings should move.

For dense test cases `CuArray(Matrix(H_cpu))` works too, but for any realistic many-body
Hamiltonian you want the sparse CSR form.

## Strategy on GPU

Use [`BlockWise`](@ref Polfed.BlockWise) — the device supplies its own parallelism and
host-side decomposition only adds overhead. It is already the default for device arrays;
setting it explicitly just documents the intent.

```julia
mc_gpu = MappingConfig(
    parallel_strategy = BlockWise(),
    optimize_mapping  = true,
)

result = polfed(H_gpu, x0_gpu, howmany, target; mc = mc_gpu)
display_report(result)
```

`optimize_mapping` works the same way as on CPU (see
[Optimized kernels](@ref optimized_kernels)).

## Custom rescaled mapping on GPU

The [rescaling advice](@ref operator_interface) applies unchanged — do the rescaling once,
in the data, not on every Clenshaw step. A practical pattern is to estimate the bounds on
CPU, build the rescaled matrix there, then move it across:

```julia
using SparseArrays

f_cpu! = (Y, X) -> mul!(Y, H_cpu, X)
Emin, Emax = Polfed.Numerics.Lanczos.lanczos_extrema(f_cpu!, x0_cpu; maxdim=1000)

spread = (Emax - Emin) / 2
center = (Emax + Emin) / 2

H_rescaled = copy(H_cpu)
H_rescaled .*= 1 / spread
H_rescaled += spdiagm(0 => fill(-center / spread, size(H_cpu, 1)))
H_gpu_rescaled = CUDA.CUSPARSE.CuSparseMatrixCSR(H_rescaled)

mc = MappingConfig(
    parallel_strategy = BlockWise(),
    f!_rescaled = (Y, X) -> mul!(Y, H_gpu_rescaled, X),
    Emin = Emin,
    Emax = Emax,
)

result = polfed(H_gpu, x0_gpu, howmany, target; mc = mc)
display_report(result)
```

## Custom Clenshaw kernels on GPU

For model-specific tuning, supply the Clenshaw steps yourself:

```julia
function make_gpu_clenshaw_kernels(H_gpu)
    crr = (b1, b2, b3, c, X) -> begin
        tmp = similar(X)
        mul!(tmp, H_gpu, b2)
        @. b1 = c * X + 2 * tmp - b3
        return nothing
    end

    cfs = (b1, b2, c, Y, X) -> begin
        tmp = similar(X)
        mul!(tmp, H_gpu, b1)
        @. Y = c * X + tmp - b2
        return nothing
    end

    return crr, cfs
end

crr_gpu!, cfs_gpu! = make_gpu_clenshaw_kernels(H_gpu)

mc = MappingConfig(
    parallel_strategy   = BlockWise(),
    optimize_mapping    = true,
    clenshaw_recurrence = crr_gpu!,
    clenshaw_finalsum   = cfs_gpu!,
)

result = polfed(H_gpu, x0_gpu, howmany, target; mc = mc)
display_report(result)
```

These are templates, not production kernels — each allocates a temporary per call, which is
the first thing to remove in a fused model-specific implementation. The integration point
stays [`MappingConfig`](@ref Polfed.MappingConfig).

## Block Lanczos helps more on GPU

A block turns the transform's matrix-**vector** products into matrix-**matrix** products,
which raises arithmetic intensity and gives the device far more concurrent work. The same
block-size rule still applies (``\texttt{howmany}/\texttt{block\_size} \gtrsim 100``) — see
[Parallelization](@ref parallelization).

```julia
x0_gpu_block = CuArray(Matrix(qr(rand(size(H_cpu, 1), 8)).Q))
result = polfed(H_gpu, x0_gpu_block, howmany, target)
```

## Reading the output

- **Mapping → processing unit** should say GPU. If it says CPU, an array did not make it to
  the device — a common cause is a CPU `x0` paired with a GPU matrix.
- **Spectral Transformation → matrix multiplications** must match the CPU run for the same
  problem. This is the check that you are comparing like with like.
- **Timings** — CPU time is deliberately **suppressed** on GPU runs, where it measures host
  bookkeeping rather than device work and would mislead. Compare wall time.

Element type matters more here than on CPU: `Float32` and `ComplexF32` are typically much
faster on consumer devices, and POLFED supports both. Set the tolerances accordingly —
`eigentol = 1e-9` is not reachable in single precision.

## Next

- [Optimized kernels](@ref optimized_kernels) — the CPU-side equivalent.
- [Guidelines](@ref guidelines) — settings by problem size.
