# MappingKernel — the user's CHOICE of row kernel, kept separate from MappingStructure, which is
# the FACTS about H.
#
#   MappingStructure  ← derived from H by `classify`  (diagonal, value groups, row lengths)
#   MappingKernel     ← chosen by the caller           (which kernel, and its parameters)
#
# Nothing about the Hamiltonian belongs here: not the dimension, not the row lengths, not which
# symmetry sector it is. Those are read off H. A kernel spec carries only what a person decides.
#
# The two symbols mirror the folder layout: `pattern` is the folder under patterns/, `variant` is
# the kernel inside its kernels/. Construction validates against the registry alone — no H needed —
# so an unknown kernel or a parameter it does not accept fails immediately and legibly.

"""
    MappingKernel(pattern, variant; params...)
    MappingKernel(pattern; params...)

The choice of row-mapping kernel: which kernel, plus its tuning parameters.

```julia
MappingKernel(:universal, :row_read; G = 8)   # pattern and variant
MappingKernel(:universal)                     # default variant, default parameters
```

Parameters you do not give are filled from the kernel's own declared defaults, so `params` always
holds the values that will actually run — printing a `MappingKernel` shows the real configuration,
not just the part you typed.

`pattern` names the off-diagonal structure the kernel targets — the folder under `patterns/` —
and is checked against [`classify`](@ref) when the kernel is built, so asking for a kernel this
matrix cannot use is reported plainly rather than failing somewhere deep.

Everything else the kernel needs — the diagonal, the off-diagonal values, the neighbour tables,
the row lengths — is derived from `H`. Pass this to [`MappingConfig`](@ref) as `optimize_mapping`.

# Fields
- `pattern::Symbol`: the structure the kernel targets.
- `variant::Symbol`: which kernel within that pattern.
- `params::NamedTuple`: the **effective** tuning parameters — the kernel's declared defaults with
  your overrides merged over them.

# Throws
- `ArgumentError` for an unknown kernel, or a parameter the kernel does not accept.
"""
struct MappingKernel
    pattern::Symbol
    variant::Symbol
    params::NamedTuple

    function MappingKernel(pattern::Symbol, variant::Symbol; params...)
        info = _find(pattern, variant)
        isnothing(info) && throw(ArgumentError(
            "MappingKernel: no kernel :$(pattern)/:$(variant). Registered: $(_kernel_names())."))
        nt = NamedTuple(params)
        for name in keys(nt)
            name in keys(info.params) || throw(ArgumentError(
                "MappingKernel: kernel :$(pattern)_$(variant) does not accept parameter `$name`; " *
                "it accepts " *
                (isempty(info.params) ? "none." : join(keys(info.params), ", ") * ".")))
        end
        # the kernel's defaults, with the caller's overrides on top
        return new(pattern, variant, merge(info.params, nt))
    end
end

function MappingKernel(pattern::Symbol; params...)
    info = _default_variant(pattern)
    isnothing(info) && throw(ArgumentError(
        "MappingKernel: no pattern :$(pattern). Registered: $(_kernel_names())."))
    return MappingKernel(pattern, info.variant; params...)
end

"""    kernel_name(k::MappingKernel) -> Symbol — the `:universal_row_read` form."""
kernel_name(k::MappingKernel) = Symbol(k.pattern, :_, k.variant)

function Base.show(io::IO, k::MappingKernel)
    print(io, "MappingKernel(:", k.pattern, ", :", k.variant)
    for (name, v) in pairs(k.params)
        print(io, ", ", name, " = ", v)
    end
    print(io, ")")
    return nothing
end

"""
    _default_kernel(s::MappingStructure, bs::Integer) -> MappingKernel

The kernel used when the caller says `optimize_mapping = true` without naming one: among the
**universal** kernels that can run on `s`, the one that most wants block size `bs`. Only that
pattern is auto-selected — see the loop body.

Two questions, kept apart. [`can_apply`](@ref) asks whether a kernel *may* run on this structure;
[`auto_score`](@ref) asks whether it *should* be chosen at this block size. A kernel that is
correct everywhere but only fast in a corner (`row_bucketed` at `bs = 1`) scores itself negative
elsewhere and is skipped, while staying available by name. Ties go to the earlier registry entry,
so `row_read` stays the default.

No parameter is named here — this layer knows nothing about any kernel's knobs, so a kernel that
declares parameters (or none) needs no change to this function.

# Throws
- `ArgumentError` if no registered kernel can run on `s`.
"""
function _default_kernel(s::MappingStructure, bs::Integer; require_range::Bool = false)
    best = nothing
    best_score = 0
    for info in REGISTRY
        # Only `universal` is wired for automatic selection. spin_flip and hopping stay fully
        # implemented and reachable BY NAME (`optimize_mapping = :spin_flip`, or a MappingKernel),
        # but `optimize_mapping = true` must not silently pick a pattern-specific kernel — on a
        # spin-flip matrix `spin_flip/packed` scores 100 against `universal/row_read`'s 0 and
        # would win outright.
        info.pattern === :universal || continue
        # A caller that owns the threads needs the kernel's SERIAL range form; a kernel without
        # one would open its own `@threads :static` inside the caller's, which is illegal. At
        # bs = 1 this is what keeps auto-selection off `row_bucketed`, whose per-bucket regions
        # cannot be expressed as one flat range.
        require_range && !supports_range(Val(info.pattern), Val(info.variant)) && continue
        can_apply(Val(info.pattern), Val(info.variant), s) || continue
        score = auto_score(Val(info.pattern), Val(info.variant), Int(bs))
        score < 0 && continue
        if isnothing(best) || score > best_score
            best, best_score = info, score
        end
    end
    isnothing(best) && throw(ArgumentError(
        "optimize_mapping: no registered kernel can run on this matrix. Registered: " *
        "$(_kernel_names()). Use optimize_mapping = false for the generic path."))
    return MappingKernel(best.pattern, best.variant)
end

"""
    build_kernel(k::MappingKernel, s::MappingStructure, ::Val{bs}) -> Function

Build the row-layout closure for `k` over structure `s`, specialized to block size `bs`.
Dispatches into the kernel's own pattern folder, splatting the effective parameters — already
complete, since [`MappingKernel`](@ref) merged the kernel's defaults at construction.

# Throws
- `ArgumentError` if `k`'s pattern cannot run on `s`.
"""
function build_kernel(k::MappingKernel, s::MappingStructure, ::Val{bs}) where {bs}
    can_apply(Val(k.pattern), Val(k.variant), s) || throw(ArgumentError(
        "build_kernel: $(k) cannot run on this matrix — " *
        "$(why_not(Val(k.pattern), Val(k.variant), s)). " *
        "Run mapping_kernels(H) to see what applies."))
    return build_kernel(Val(k.pattern), Val(k.variant), s, Val(bs); k.params...)
end

"""
    build_range_kernel(k::MappingKernel, s::MappingStructure, ::Val{bs}) -> Function

The SERIAL range closure `(Y, X, lo, hi) -> nothing` for `k` over work units `lo:hi` (1-based,
inclusive). Opens no parallel region, so a caller doing its own decomposition can call it from
inside its own — which [`build_kernel`](@ref)'s closure cannot, because `Threads.@threads :static`
does not nest.

Pair with [`range_units`](@ref) (how many units) and [`range_alignment`](@ref) (what the chunk
boundaries must be multiples of). Check [`supports_range`](@ref) first: not every kernel has one.

# Throws
- `ArgumentError` if `k`'s pattern cannot run on `s`, or if `k` has no range form.
"""
function build_range_kernel(k::MappingKernel, s::MappingStructure, ::Val{bs}) where {bs}
    can_apply(Val(k.pattern), Val(k.variant), s) || throw(ArgumentError(
        "build_range_kernel: $(k) cannot run on this matrix — " *
        "$(why_not(Val(k.pattern), Val(k.variant), s))."))
    return build_range_kernel(Val(k.pattern), Val(k.variant), s, Val(bs); k.params...)
end

"""    range_units(k::MappingKernel, s::MappingStructure, ::Val{bs}) -> Int — see [`build_range_kernel`](@ref)."""
range_units(k::MappingKernel, s::MappingStructure, ::Val{bs}) where {bs} =
    range_units(Val(k.pattern), Val(k.variant), s, Val(bs); k.params...)

"""    range_alignment(k::MappingKernel, s::MappingStructure, ::Val{bs}) -> Int — see [`build_range_kernel`](@ref)."""
range_alignment(k::MappingKernel, s::MappingStructure, ::Val{bs}) where {bs} =
    range_alignment(Val(k.pattern), Val(k.variant), s, Val(bs); k.params...)

"""    supports_range(k::MappingKernel) -> Bool — whether `k` exposes a serial range form."""
supports_range(k::MappingKernel) = supports_range(Val(k.pattern), Val(k.variant))
