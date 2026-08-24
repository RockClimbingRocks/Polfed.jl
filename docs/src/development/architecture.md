# Architecture and development

UniPolfed separates reusable numerical kernels from the POLFED workflow and from
the physical models used to construct Hamiltonians. The package root loads the
components in dependency order:

```text
UniPolfed
├── common          logging, formatting, timing, numerical utilities
├── backend         CPU/GPU dispatch hooks and processing-unit types
├── numerics
│   ├── clenshaw    polynomial evaluation
│   ├── kpm         density-of-states primitives
│   └── lanczos     scalar and block Lanczos
├── models          QSun, XXZ, and J1-J2 Hamiltonians
├── optimization    row-layout mappings, kernel registry, benchmarks
└── main
    ├── defaults
    ├── logic       mapping, DoS, targets, filters, transforms, reports
    ├── pipeline    single-window, patched, and full-spectrum workflows
    └── entrypoint  public `polfed`, `patched_polfed`, and `full_polfed`
```

## Module boundaries

`Common`, `Numerics`, `Models`, and `Optimization` are submodules.
Files below `main/` are included directly into `UniPolfed`, so the public
algorithm types and functions live at the package top level. `backend/` is also
included at the top level because the CUDA extension installs methods into
`UniPolfed`'s dispatch hooks.

An included file shares the namespace of its enclosing module. External
packages therefore belong in the module root that uses them, before its
`include` statements. Imports between submodules should be relative. A sibling
is reached with `..Sibling`; a name owned by `UniPolfed` from a nested module
requires enough leading dots to reach the package root.

Dependencies should point inward toward reusable layers:

- `numerics/` must not depend on POLFED pipeline types.
- `models/` constructs operators without knowing how they will be diagonalized.
- `optimization/` can be tested independently from the public pipeline.
- `main/` composes all lower layers.

## Runtime data flow

A normal run follows this sequence:

1. `MappingConfig` resolves spectral bounds and the affine map to `[-1, 1]`.
2. KPM estimates the density of states on the rescaled operator.
3. A target becomes a rescaled interval.
4. `resolve_filter` selects and validates a filter for that interval.
5. Clenshaw builds the transformed operator.
6. Lanczos selects eigenpairs at the exposed transformed end.
7. `PolfedResult` maps values back to physical energy, trims them, and records
   reports and timings.

Configuration objects describe user choices. A separate plan is introduced only
when setup resolves derived runtime state; `MappingConfig` has a `MappingPlan`,
whereas `DoSConfig` is already concrete and needs no plan.

## Numerical invariants

- Lanczos returns a contiguous converged run from each requested transformed
  extreme. Every returned pair must satisfy the applicable convergence gates.
- A `Window` run may retain one bracketing pair on each finite side. Those
  brackets certify enclosure; they are not part of the requested in-window
  spectrum.
- Column layout stores a block as `D × blocksize`; row layout stores it as
  `blocksize × D`. Code that reports capacity or slices eigenvectors must use
  the layout's vector axis.
- KPM moments operate on a rescaled operator whose spectrum lies in `[-1, 1]`.
- Pipeline timing uses named `StageTimer` stages instead of positional slots.

## Filters and mappings

Each filter owns its type, coefficients, construction, and validation in one
folder under `main/logic/filter/filter_types/`. Shared code talks to filters through
the `AbstractFilter` interface. Interior windows use delta filters; spectral-edge
windows use a monotone `EdgeFilter` that cannot develop side lobes.

The matrix entrypoint can replace generic sparse multiplication with row-layout
kernels. `benchmark_mappings` should select kernels on the actual machine and
block size; performance rankings are hardware- and model-dependent. The generic
`mul!` implementation remains the correctness baseline.

## Naming and testing

- Folders and action-oriented files use `snake_case`.
- Files centered on a type or module use `PascalCase` matching that type.
- Tests mirror `src/` and assert mathematical ground truth rather than rendered
  snapshots.
- Cover `Float32`, `Float64`, and complex element types where the algorithm
  supports them, with type-appropriate tolerances.
- Parametrize hot-path callbacks and keep struct fields concrete.
- Do not index task-local buffers with `Threads.threadid()`; Julia tasks can
  migrate. Prefer chunk-owned accumulators and reduce them afterward.

From the repository root:

```bash
julia --project=. -e 'using UniPolfed'
julia --project=. -e 'using Pkg; Pkg.test()'
bash docs/build.sh
julia --project=. examples/reporting.jl
```

CUDA behavior is provided by `ext/UniPolfedCUDAExt.jl` and is testable only on a
CUDA-capable host.
