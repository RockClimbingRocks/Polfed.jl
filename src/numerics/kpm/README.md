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

There are no KPM-specific layout types, batching options, or operator convenience
methods in this version. If `f!` expects row storage, pass a row-shaped template.
If it expects column storage, pass a column-shaped template.
