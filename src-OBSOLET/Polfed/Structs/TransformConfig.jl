"""
    TransformConfig(; kwargs...)

Configuration for polynomial spectral transformation used by [`polfed`](@ref).

This struct controls how the filter polynomial is constructed around the chosen
target: coefficients, normalization/cutoff behavior, interval constraints, and
order selection.

Pass this struct as `transform=...` in [`polfed`](@ref).

# Fields
- `coefficients::Function`
  Coefficient generator used by Clenshaw polynomial evaluation.
  Expected callable form:
  `coefficients(target_rescaled::Real, n::Int) -> Real`.
  The default is Chebyshev-like coefficients from
  [`PolfedDefaults.coefficients`](@ref).

- `normalization::Real`
  Output normalization of the polynomial filter.
  POLFED rescales the polynomial so that `P(target) = normalization`.

- `cutoff::Real`
  Threshold used when deriving interval width/order automatically.
  Lower values typically imply wider filters and/or larger polynomial orders.

- `left::Union{Real,Nothing}`, `right::Union{Real,Nothing}`
  Optional *unrescaled* interval bounds.
  If provided, both should be set together.
  They are internally converted to rescaled coordinates.

- `order::Union{Int,Nothing}`
  Optional fixed polynomial order `K`.
  If `nothing`, POLFED computes `K` from the chosen interval and `cutoff`.

- `order_safety_factor::Real`
  Multiplicative safety factor used when order is computed automatically.
  Values below `1` reduce order (and POLFED recomputes a consistent interval).
  This is often used to improve robustness when requesting many eigenpairs.

- `polynomialtype::Symbol`
  Polynomial family used by Clenshaw evaluation.
  Supported types follow internal Clenshaw kernels (for example
  `:Chebyshev`, `:Legendre`, `:Hermite`, `:Taylor`).

# Keyword Defaults
- `coefficients = PolfedDefaults.coefficients`
- `normalization = PolfedDefaults.normalization`
- `cutoff = PolfedDefaults.cutoff`
- `left = nothing`
- `right = nothing`
- `order = nothing`
- `order_safety_factor = PolfedDefaults.order_safety_factor`
- `polynomialtype = PolfedDefaults.polynomialtype`

# Notes
- Do not set both explicit bounds (`left`/`right`) and explicit `order`:
  this overconstrains the filter and raises an error.
- Plain numeric `target` or `(:unrescaled, E)` is converted to rescaled
  coordinates internally before calling `coefficients`.
- If you use custom coefficients/polynomial families, consider validating
  `order`, interval, and convergence with `produce_report=true`.

# Example
```julia
transform = TransformConfig(
    cutoff = 0.17,
    order_safety_factor = 0.95,
)
vals, vecs, report = polfed(mat, x0, howmany, target;
    produce_report = true,
    transform = transform,
)
```
"""
mutable struct TransformConfig
    coefficients::Function
    normalization::Real
    cutoff::Real
    left::Union{Real, Nothing}
    right::Union{Real, Nothing}
    order::Union{Int, Nothing}
    order_safety_factor::Real
    polynomialtype::Symbol

    """Build a `TransformConfig` from keyword arguments."""
    function TransformConfig(;
        coefficients::Function        = PolfedDefaults.coefficients,
        normalization::Real           = PolfedDefaults.normalization,
        cutoff::Real                  = PolfedDefaults.cutoff,
        left::Union{Real, Nothing}    = nothing,
        right::Union{Real, Nothing}   = nothing,
        order::Union{Int, Nothing}    = nothing,
        order_safety_factor::Real     = PolfedDefaults.order_safety_factor,
        polynomialtype::Symbol        = PolfedDefaults.polynomialtype,
    )
        new(coefficients, normalization, cutoff, left, right, order, order_safety_factor, polynomialtype)
    end
end
