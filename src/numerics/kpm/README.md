# Kernel Polynomial Method (KPM)

Direction-independent KPM moments and density-of-states reconstruction. The module is
loaded by `Numerics.jl` and used by UniPolfed's density-of-states stage.

The public API is intentionally small:

```julia
moments = kpm_moments(f!, N, R, template, CPU())
rho = kpm_density(moments; kernel = :Jackson)
```

`template` tells KPM what shape to allocate and pass into `f!`.

## Column-vector action

```julia
template = Vector{Float64}(undef, size(H, 1))
f!(y, x) = (mul!(y, H, x); nothing)

mu = kpm_moments(f!, 250, 300, template, CPU())
```

## Column-matrix action

```julia
template = Matrix{Float64}(undef, size(H, 1), 1)
f!(Y, X) = (mul!(Y, H, X); nothing)

mu = kpm_moments(f!, 250, 300, template, CPU())
```

## Row-matrix action

```julia
template = Matrix{Float64}(undef, 1, size(H, 1))
f!(Y, X) = (mul!(Y, X, H); nothing)

mu = kpm_moments(f!, 250, 300, template, CPU())
```

There are no KPM-specific layout types or operator convenience methods in this
version. If `f!` expects row storage, pass a row-shaped template. If it expects column
storage, pass a column-shaped template. Keep the block axis shorter than `dim` — the larger
axis is read as the Hilbert space, and a square template as columns.

## Probes, blocks and tasks

A matrix template applies several probes per `f!` call, which reads the operator once for the
whole block. The probes are still independent:

- each probe is drawn on its own (`randn!` into a length-`dim` buffer, then placed in its
  slot), so probe `r` is the `r`-th draw from `rng` whatever the block width;
- each probe's moments are kept in their own column (`probe_moments!`) and summed in probe
  order `1…R` at the end.

`ntasks` runs that many blocks concurrently, each with its own buffers; the draws are
serialized in probe order, so the result is bit-identical for every `ntasks`. `f!` must be
safe to call concurrently on distinct arrays. Library threading is the caller's business —
UniPolfed's DoS stage wraps the call in `with_serial_libraries`.

```julia
mu = kpm_moments(f!, 250, 300, zeros(8, size(H, 1)), CPU(); rng = MersenneTwister(1),
                 ntasks = Threads.nthreads())
```
