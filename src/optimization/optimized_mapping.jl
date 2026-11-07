# The boundary the POLFED pipeline consumes. `_build_matrix_ops` calls exactly one function
# here — `build_optimized_mapping` — so everything above this line is free to change shape.
#
# `resolve_mapping_kernel` turns whatever the user put in `optimize_mapping` into a concrete
# `MappingKernel`; `build_optimized_mapping` then builds it, VERIFIES it against `mul!`, and hands
# back the closure.

"""
    resolve_mapping_kernel(choice, s::MappingStructure, bs::Integer) -> MappingKernel

Turn an `optimize_mapping` request into a concrete kernel for the matrix described by `s`, at
Krylov block size `bs`.

| `choice` | meaning |
|:---|:---|
| `true`, `:auto` | let POLFED pick — see [`_default_kernel`](@ref); the choice depends on `bs` |
| `:universal` | that pattern's default variant |
| `:universal_row_read` | that exact kernel |
| [`MappingKernel`](@ref) | pin the kernel and its parameters |
| `(kernel = …, G = …)` named tuple | legacy spelling, converted to a `MappingKernel` |

`false` never reaches here — the caller takes the generic path instead.

# Throws
- `ArgumentError` if the request names an unknown kernel, or one that cannot run on `s`.
"""
function resolve_mapping_kernel(choice, s::MappingStructure, bs::Integer;
                                require_range::Bool = false)
    k = _as_kernel(choice, s, bs; require_range = require_range)
    can_apply(Val(k.pattern), Val(k.variant), s) || throw(ArgumentError(
        "resolve_mapping_kernel: $(k) cannot run on this matrix — " *
        "$(why_not(Val(k.pattern), Val(k.variant), s)). " *
        "Run mapping_kernels(H) to see what applies."))
    return k
end

_as_kernel(k::MappingKernel, ::MappingStructure, ::Integer; require_range::Bool = false) = k

_as_kernel(choice::Bool, s::MappingStructure, bs::Integer; require_range::Bool = false) =
    choice ? _default_kernel(s, bs; require_range = require_range) :
             throw(ArgumentError("resolve_mapping_kernel: `false` takes the generic path."))

function _as_kernel(choice::Symbol, s::MappingStructure, bs::Integer; require_range::Bool = false)
    choice === :auto && return _default_kernel(s, bs; require_range = require_range)
    # a bare pattern name — `:universal`
    isnothing(_default_variant(choice)) || return MappingKernel(choice)
    # `pattern_variant` — split at each underscore and take the first split that resolves, so a
    # variant containing an underscore (`:row_read`) still works
    name = String(choice)
    for (i, c) in enumerate(name)
        c == '_' || continue
        info = _find(Symbol(name[1:i-1]), Symbol(name[i+1:end]))
        isnothing(info) || return MappingKernel(info.pattern, info.variant)
    end
    throw(ArgumentError(
        "resolve_mapping_kernel: unknown kernel :$(choice). Registered: $(_kernel_names())."))
end

function _as_kernel(choice::NamedTuple, s::MappingStructure, bs::Integer; require_range::Bool = false)
    haskey(choice, :kernel) || throw(ArgumentError(
        "resolve_mapping_kernel: a named-tuple request must carry `kernel`, got $(keys(choice))."))
    rest = Base.structdiff(choice, NamedTuple{(:kernel,)})
    k = _as_kernel(choice.kernel, s, bs; require_range = require_range)
    return MappingKernel(k.pattern, k.variant; rest...)
end

_as_kernel(choice, ::MappingStructure, ::Integer; require_range::Bool = false) = throw(ArgumentError(
    "resolve_mapping_kernel: cannot interpret optimize_mapping = $(repr(choice)). Use `true`, " *
    "a kernel name, or a MappingKernel."))

"""
    build_optimized_mapping(H, choice, ::Val{bs}, a, b) -> (; f!_rescaled, kernel, structure)

Build the row-layout action of the **rescaled** operator ``\\tilde H = (H - b)/a`` for the POLFED
transform. This is the single entry point the pipeline uses.

```
classify(H) → resolve the kernel → rescale the structure → build → verify against mul!
```

The verification is not optional: a row kernel walks index tables by hand, and a mistake there
returns a slightly wrong matvec rather than an error, which POLFED would happily converge on. One
block checked against `mul!` costs two matvecs against the thousands a run performs. See
[`verify_kernel`](@ref).

# Arguments
- `H`: the Hamiltonian; any structured matrix `classify` accepts.
- `choice`: the `optimize_mapping` request — see [`resolve_mapping_kernel`](@ref).
- `bs`: Krylov block size, as a `Val` so the kernel specializes on it.
- `a`, `b`: the rescaling, ``\\tilde H = (H - b)/a``.
- `serial`: build the kernel's SERIAL range form over the whole domain instead of the
  self-threading whole-block form. Required when the caller opens its own parallel region —
  [`VectorWise`](@ref) runs one task per Krylov vector, and a kernel opening
  `Threads.@threads :static` inside that is illegal. Restricts auto-selection to kernels that
  have a range form (see [`supports_range`](@ref)).

# Returns
A named tuple `(; f!_rescaled, kernel, structure)`.
"""
function build_optimized_mapping(H::AbstractMatrix, choice, ::Val{bs}, a::Real, b::Real;
                                 serial::Bool = false) where {bs}
    s = classify(H)
    kernel = resolve_mapping_kernel(choice, s, bs; require_range = serial)
    sr = rescale(s, a, b)
    f!_rescaled = if serial
        supports_range(kernel) || throw(ArgumentError(
            "build_optimized_mapping: $(kernel) has no serial range form, so it cannot be built " *
            "for a caller that owns the threads. Name a kernel with one, or use a parallelization " *
            "that lets the kernel thread itself."))
        fr! = build_range_kernel(kernel, sr, Val(bs))
        n = range_units(kernel, sr, Val(bs))
        (Y, X) -> (fr!(Y, X, 1, n); nothing)
    elseif supports_range(kernel)
        fr! = build_range_kernel(kernel, sr, Val(bs))
        n = range_units(kernel, sr, Val(bs))
        align = range_alignment(kernel, sr, Val(bs))
        # the transform will call the range form chunk by chunk: check it across a chunk boundary
        verify_kernel(_split_range(fr!, n, align), H, a, b, Val(bs), kernel_name(kernel))
        RangeKernelOperator(build_kernel(kernel, sr, Val(bs)), fr!, n, align)
    else
        build_kernel(kernel, sr, Val(bs))
    end
    verify_kernel(f!_rescaled, H, a, b, Val(bs), kernel_name(kernel))
    return (; f!_rescaled, kernel, structure = s)
end

"""
    RangeKernelOperator(whole, range!, units, alignment)

A built kernel that also carries its serial range form ([`build_range_kernel`](@ref)). Callable
as the whole-block operator `op(Y, X)`; the transform uses `range!`/`units`/`alignment` to run the
whole Clenshaw recurrence in one threaded region ([`RangeClenshaw`](@ref)) instead of opening the
kernel's region and a combine region on every step.
"""
struct RangeKernelOperator{W,R} <: Function
    whole::W
    range!::R
    units::Int
    alignment::Int
end

(op::RangeKernelOperator)(Y, X) = (op.whole(Y, X); nothing)

# The range form applied as two chunks split on an aligned boundary.
_split_range(fr!, n::Int, align::Int) = (Y, X) -> begin
    m = min(n, cld(cld(n, align), 2) * align)
    fr!(Y, X, 1, m)
    m < n && fr!(Y, X, m + 1, n)
    return nothing
end
