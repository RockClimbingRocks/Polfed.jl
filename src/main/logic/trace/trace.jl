# Trace slice: the composite log lines that span stages. Everything a single stage knows about
# itself is logged inside that stage (mapping/, dos/, target/, interval/, filter/, transform/);
# what lands here is the cross-slice narrative — what the run was asked for, the full parameter
# set handed to Lanczos, and what came back — so that all four launch sites (the two single-filter
# pipelines, the patched patch runner, the double runner) print in ONE shape rather than each
# growing its own.
#
# Everything here is log-only: no return value feeds back into the algorithm, and every payload
# that costs more than a field read sits behind `should_log`.

"""
    _describe_end(spec::EndSpec) -> String

One-line rendering of a Lanczos end-spec ([`NumEigs`](@ref)/[`ToValue`](@ref)/[`Window`](@ref)) —
the *stop rule* the run is actually given, which no report field exposes before the run.
"""
_describe_end(spec::NumEigs) = "NumEigs($(spec.n))"
_describe_end(spec::ToValue) = @sprintf("ToValue(%.6g)", spec.value)
_describe_end(spec::Window) = @sprintf(
    "Window(%.6g, %.6g; edge=%s, brackets=%s)", spec.low, spec.high,
    spec.edge_value === nothing ? "none" : @sprintf("%.4g", spec.edge_value), spec.return_brackets)
_describe_end(spec) = string(spec)

"""
    describe_eig_target(target) -> String

One-line rendering of a composed [`EigTarget`](@ref) (`Lowest`/`Highest`/`BothEnds` over its
end-specs) for the pre-Lanczos trace.
"""
describe_eig_target(target::Lowest)  = "Lowest("  * _describe_end(target.spec) * ")"
describe_eig_target(target::Highest) = "Highest(" * _describe_end(target.spec) * ")"
describe_eig_target(target::BothEnds) =
    "BothEnds(low=" * _describe_end(target.low) * ", high=" * _describe_end(target.high) * ")"
describe_eig_target(target) = string(target)

"""`_request_of(filter)` — the window a filter was resolved for (the hull, for a two-peak filter)."""
_request_of(filter::AbstractFilter) = filter.interval
_request_of(filter::DoubleDeltaFilter) = Interval(filter.interval1.left, filter.interval2.right)

"""
    _window_text(filter, interval, mp) -> String

The exposed window in both coordinate systems. A [`DoubleDeltaFilter`](@ref) exposes TWO windows
with a suppressed valley between them, so printing their hull would claim a region the run does
not target — it renders both instead.
"""
_window_text(::AbstractFilter, interval::Interval, mp::MappingPlan) =
    "$(_fmt_iv(interval)) rescaled, $(_fmt_iv(interval, mp.a, mp.b)) in energy"
_window_text(filter::DoubleDeltaFilter, ::Interval, mp::MappingPlan) =
    "$(_fmt_iv(filter.interval1)) ∪ $(_fmt_iv(filter.interval2)) rescaled, " *
    "$(_fmt_iv(filter.interval1, mp.a, mp.b)) ∪ $(_fmt_iv(filter.interval2, mp.a, mp.b)) in energy"

"""
    log_run_start(routine, mode, spec; kwargs...) -> nothing

The one-line INFO header opening a pipeline: which entrypoint, which mode, and the target as the
user named it. Extra `kwargs` (counts, tolerances) are appended verbatim.
"""
function log_run_start(routine::AbstractString, mode::Symbol, spec::TargetSpec; kwargs...)
    should_log(POLFED_INFO_LEVEL) || return nothing
    polfed_log(POLFED_INFO_LEVEL, "$routine started ($mode mode).";
               target = format_target_strategy(spec), kwargs...)
    return nothing
end

"""
    log_prelanczos(mp, filter, eig_target, howmany, expected_dim, maxdim, tol, eigentol;
                   interval, label, rescaled_values) -> nothing

Everything the filtered Lanczos run is about to be given, in one place — the last point where a
run can still be understood before it turns into iterations.

INFO prints what the run *is*: the resolved filter, the exposed window in both coordinate systems,
the [`EigTarget`](@ref) stop rule, the Krylov budget, and the projected `H·v` cost (the spectral
transformation is POLFED's bottleneck, so `K·blocksize` per iteration is the number that explains
a slow run). DEBUG adds the parameters that decide whether the run is even asking the right
question: which operator the Ritz values are measured against, the tolerances *in the units they
are applied in*, and how much wider than the request the passband is.

One condition is worth a warning here and nowhere else, because this is where the request and the
resolved geometry are both known: a passband so much wider than the request that the run will
spend its budget converging pairs nobody asked for.

# Arguments
- `mp::MappingPlan`: for `dim`/`block_size` and the rescaling used to show true energies.
- `filter::AbstractFilter`, `eig_target`: the resolved filter and the Lanczos stop rule.
- `howmany`, `expected_dim`, `maxdim`: the requested count and the Krylov budget around it.
- `tol`, `eigentol`: the Ritz and eigen tolerances as passed to `lanczos`.

# Keyword Arguments
- `interval::Interval`: the requested window (defaults to the one the filter was resolved for).
- `label::AbstractString`: run identity, e.g. `"POLFED"` or `"Interior 3/17"`.
- `rescaled_values::Bool`: `true` when `mapvals` produces rescaled energies (so the tolerances and
  the returned values live in `H̃` units until the result boundary un-rescales them).
"""
function log_prelanczos(mp::MappingPlan, filter::AbstractFilter, eig_target,
                        howmany::Integer, expected_dim::Integer, maxdim::Integer,
                        tol::Real, eigentol::Real;
                        interval::Interval = _request_of(filter),
                        label::AbstractString = "POLFED",
                        rescaled_values::Bool = false)
    K, bs, band = polynomial_order(filter), mp.sys.block_size, passband(filter)
    width_ratio = isfinite(band.left) && isfinite(band.right) ?
        (band.right - band.left) / max(interval.right - interval.left, eps()) : NaN

    width_ratio > Defaults.polfed.passband_warn_ratio && polfed_log(POLFED_WARN_LEVEL,
        "The filter's passband is much wider than the requested window, so the run will converge " *
        "many more eigenpairs than were asked for before it can stop.",
        passband = _fmt_iv(band), request = _fmt_iv(interval), width_ratio = width_ratio)

    should_log(POLFED_INFO_LEVEL) || return nothing
    polfed_log(POLFED_INFO_LEVEL, "$label: launching Lanczos on the filtered operator.",
        filter = _filter_tag(filter),
        window = _window_text(filter, interval, mp),
        eig_target = describe_eig_target(eig_target),
        krylov = "expected $expected_dim, maxdim $maxdim, blocksize $bs, howmany $howmany",
        cost = "$(K * bs) H·v per Lanczos iteration, ≈ $(K * bs * expected_dim) projected")
    polfed_log(POLFED_DEBUG_LEVEL, "$label: Lanczos parameters.",
        ritz_values = rescaled_values ? "rescaled H̃ (un-rescaled at the result boundary)" :
                                        "true energies of H",
        tol = tol, eigentol = eigentol,
        eigentol_meaning = rescaled_values ?
            @sprintf("‖H·v − λ·v‖ < %.3g (= eigentol·a)", eigentol * mp.a) :
            @sprintf("‖H·v − λ·v‖ < %.3g", eigentol),
        passband_over_request = width_ratio, exposed_end = exposed_end(filter))
    return nothing
end

"""
    log_lanczos_outcome(result::LanczosResult; label) -> nothing

Close a filtered Lanczos run at INFO: how many pairs it certified, at what cost, and whether the
stop rule was actually met (`complete`) or the budget simply ran out. The residuals are the ones
the run itself measured, in the units it ran in.
"""
function log_lanczos_outcome(result::LanczosResult; label::AbstractString = "POLFED")
    should_log(POLFED_INFO_LEVEL) || return nothing
    report = result.report
    polfed_log(POLFED_INFO_LEVEL, "$label: Lanczos finished.",
        converged = result.converged, complete = report.complete,
        iterations = report.itersneeded, krylovdim = isempty(report.krylovbasisdim_history) ?
            0 : last(report.krylovbasisdim_history),
        checks = report.numofchecks, ritz_residual = report.residual,
        eigen_residual = report.eigenresidual)
    return nothing
end
