# Mapping section of the POLFED report: what operator/rescaling/parallelization the run
# actually used. A lightweight, plain-data snapshot of the resolved MappingPlan (no
# callbacks stored) so the report is cheap to hold and type-stable to display.

"""
    MappingReport

Immutable snapshot of POLFED's **mapping** stage for the report: which operator/kernel was
applied, the rescaling `H̃ = (H - center)/spread` that put the spectrum in `[-1, 1]`, and the
Krylov layout / block size / dimension / parallelization the run used.

Built by [`MappingReport`](@ref)`(mp, layout)` from the resolved [`MappingPlan`](@ref) and the
[`Layout`](@ref) Lanczos ran in. Rendered as the "Mapping" section by [`Report`](@ref)'s `show`.

# Fields
- `optimize_mapping::Union{Bool,Symbol,NamedTuple,MappingKernel}`: the mapping-optimization request carried on
  the plan (`false` → generic operator; a kernel `Symbol`/`NamedTuple` → value-grouped kernel).
- `center::Float64`, `spread::Float64`: the rescaling shift/scale (`b`/`a` on the plan), so
  `H̃ = (H - center)/spread`.
- `Emin::Float64`, `Emax::Float64`: spectral bounds used for the rescaling.
- `layout::String`: the Krylov block layout (`"ColumnLayout"` / `"RowLayout"`).
- `block_size::Int`, `dim::Int`: Krylov block size and Hilbert-space dimension (`opdim`).
- `parallel::String`: name of the resolved parallelization strategy.
"""
struct MappingReport
    optimize_mapping::Union{Bool,Symbol,NamedTuple,MappingKernel}
    center::Float64
    spread::Float64
    Emin::Float64
    Emax::Float64
    layout::String
    block_size::Int
    dim::Int
    parallel::String
end

"""
    MappingReport(mp::MappingPlan, layout::Layout) -> MappingReport

Snapshot the resolved [`MappingPlan`](@ref) (and the [`Layout`](@ref) Lanczos ran in) into the
plain-data [`MappingReport`](@ref) the report displays. Only scalars/strings are copied — the
operator closures on the plan are not retained.
"""
MappingReport(mp::MappingPlan, layout::Layout) = MappingReport(
    mp.optimize_mapping,
    Float64(mp.b), Float64(mp.a),
    Float64(mp.Emin), Float64(mp.Emax),
    string(nameof(typeof(layout))),
    mp.sys.block_size, mp.sys.dim,
    string(nameof(typeof(mp.parallel_strategy))),
)

"""
    _operator_str(optimize_mapping) -> String

Human-readable description of the operator the mapping applied, from the
`optimize_mapping` request: `false` → a generic `mul!`/user operator; `true`/`:auto` → an
auto-selected value-grouped kernel; a [`MappingKernel`](@ref) → that kernel and its parameters;
a kernel `Symbol`/`NamedTuple` → that named kernel.
"""
function _operator_str(om)
    om === false && return "generic operator (mul! / user f!)"
    (om === true || om === :auto) && return "value-grouped kernel (auto)"
    om isa MappingKernel && return "value-grouped kernel :$(kernel_name(om))" *
        (isempty(om.params) ? "" : " (" * join(("$k = $v" for (k, v) in pairs(om.params)), ", ") * ")")
    om isa Symbol && return "value-grouped kernel :$(om)"
    om isa NamedTuple && return "value-grouped kernel :$(get(om, :kernel, :auto))"
    return string(om)
end

"""
    render_mapping(io::IO, f::Formatter, mr::MappingReport) -> Nothing

Print the "Mapping" section of the report to `io`, coloring values via `f`.
"""
function render_mapping(io::IO, f::Formatter, mr::MappingReport)
    _section_header(io, f, "Mapping")
    println(io, "- operator: ", cyan(f, _operator_str(mr.optimize_mapping)))
    println(io, "- layout: ", cyan(f, mr.layout), "    block size ", cyan(f, string(mr.block_size)),
            "    opdim ", cyan(f, string(mr.dim)))
    println(io, "- parallel strategy: ", cyan(f, mr.parallel))
    # The rescaling H̃ = (H − center)/spread is fully determined by these two, so only the
    # physically meaningful bounds are printed (center = (Emax+Emin)/2, spread = (Emax−Emin)/2).
    println(io, "- Energy: Emin = ", cyan(f, @sprintf("%.3f", mr.Emin)),
            "   Emax = ", cyan(f, @sprintf("%.3f", mr.Emax)))
    return nothing
end
