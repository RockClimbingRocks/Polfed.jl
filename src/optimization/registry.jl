# The kernel registry: machine-readable, not documentation.
#
# Each entry names a kernel by (pattern, variant) — the folder under patterns/ and the file inside
# its kernels/ — and declares which parameters that kernel accepts. Those two facts are what let
# `MappingKernel` reject a parameter a kernel has no use for, at construction, rather than deep
# inside a generated loop; and what let `mapping_kernels(H)` list what is available.
#
# Adding a kernel = one file under patterns/<pattern>/kernels/ + one `build_kernel` method + one
# entry here. The seam is deliberately as narrow as the filter chain's.

"""
    KernelInfo

One registry entry.

# Fields
- `pattern::Symbol`: the folder under `patterns/` — the off-diagonal structure the kernel targets.
- `variant::Symbol`: which kernel within that pattern.
- `params::NamedTuple`: the parameters this kernel accepts, WITH their defaults — taken from the
  kernel's own declaration, never written out again here. Empty when a kernel has no knobs, which
  is the case for both kernels today.
- `applies::String`: one line for humans, shown by [`mapping_kernels`](@ref).
"""
struct KernelInfo
    pattern::Symbol
    variant::Symbol
    params::NamedTuple
    applies::String
end

"""
    REGISTRY

Every kernel POLFED can build.
"""
const REGISTRY = (
    KernelInfo(:spin_flip, :packed, PACKED_PARAMS,
               "H = diag + Σ vₖ·X^maskₖ on the full 2^L space; real eltype, power-of-two bs"),
    KernelInfo(:spin_flip, :rows, ROWS_PARAMS,
               "same spin-flip operators, any bs and any eltype — the flip fallback"),
    KernelInfo(:hopping, :packed, HOP_PACKED_PARAMS,
               "H = diag + t·Σ periodic nearest-neighbour bonds; real eltype, bs | pack width"),
    KernelInfo(:hopping, :rows, HOP_ROWS_PARAMS,
               "same lattice, any bs and any eltype — the hopping fallback"),
    KernelInfo(:universal, :row_read, NamedTuple(),
               "any structured H — several off-diagonal values and/or rows of differing length"),
    KernelInfo(:universal, :row_bucketed, NamedTuple(),
               "single-value H; rows bucketed by length — auto-selected at block size 1"),
)

"""    _find(pattern, variant) -> KernelInfo or nothing."""
function _find(pattern::Symbol, variant::Symbol)
    i = findfirst(k -> k.pattern === pattern && k.variant === variant, REGISTRY)
    return isnothing(i) ? nothing : REGISTRY[i]
end

"""    _default_variant(pattern) -> KernelInfo or nothing — a pattern's first registered kernel."""
function _default_variant(pattern::Symbol)
    i = findfirst(k -> k.pattern === pattern, REGISTRY)
    return isnothing(i) ? nothing : REGISTRY[i]
end

"""    _kernel_names() -> String — every registered kernel, for error messages."""
_kernel_names() = join((string(k.pattern, "_", k.variant) for k in REGISTRY), ", ")

"""
    mapping_kernels([io], H) -> nothing

Print the structure of `H` and the kernels that can run on it.

The header is what [`classify`](@ref) found — dimension, off-diagonal values, row-length range —
followed by one line per registered kernel, marked ✓ when its pattern accepts this structure and
✗ with the reason when it does not.

```julia
mapping_kernels(H)
```
"""
mapping_kernels(H::AbstractMatrix) = mapping_kernels(stdout, H)

function mapping_kernels(io::IO, H::AbstractMatrix)
    s = classify(H)
    show(io, MIME"text/plain"(), s)
    println(io, "\n")
    @printf(io, " %-3s %-22s %-8s %s\n", "", "kernel", "params", "applies")
    println(io, " ", "─"^76)
    for info in REGISTRY
        ok = can_apply(Val(info.pattern), Val(info.variant), s)
        note = ok ? info.applies : why_not(Val(info.pattern), Val(info.variant), s)
        params = isempty(info.params) ? "—" :
                 join(("$k=$v" for (k, v) in pairs(info.params)), ", ")
        @printf(io, " %-3s %-22s %-8s %s\n", ok ? "✓" : "✗",
                string(info.pattern, "_", info.variant), params, note)
    end
    return nothing
end
