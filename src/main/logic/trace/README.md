# `main/logic/trace/` — the run trace

The composite log lines that span stages. Each stage logs what it alone knows (`mapping/`, `dos/`,
`target/`, `interval/`, `filter/`, `transform/`, `report/`); what lives here is the cross-slice
narrative, so that all four Lanczos launch sites — the two single-filter pipelines, the patched
patch runner (`_run_filtered_patch`), the double runner (`_double_lanczos`) — print in **one**
shape instead of each growing its own.

Everything here is log-only. No return value feeds back into the algorithm, and any payload that
costs more than a field read sits behind `should_log`, so a level-1 run pays nothing for it.

## What it provides

| Function | Level | Says |
| :--- | :--- | :--- |
| `log_run_start(routine, mode, spec; …)` | INFO | which entrypoint, which mode, the target as the user named it |
| `log_prelanczos(mp, filter, eig_target, …)` | INFO + DEBUG | the whole setup handed to Lanczos (below) |
| `log_lanczos_outcome(result; label)` | INFO | converged count, completeness, iterations, residuals |
| `describe_eig_target(target)` | — | one-line rendering of a composed `EigTarget` |

## The pre-Lanczos summary

The last point at which a run can still be *understood* rather than watched. INFO prints what the
run is — resolved filter and order, the exposed window in rescaled **and** true-energy
coordinates, the `EigTarget` stop rule, the Krylov budget, and the projected `H·v` cost
(`K · blocksize` per Lanczos iteration: the spectral transformation is POLFED's bottleneck, so
this is the number that explains a slow run). DEBUG adds what decides whether the run is asking
the right question at all: which operator the Ritz values are measured against, the tolerances in
the units they are applied in (`eigentol / a` for a rescaled window run), and how much wider than
the request the passband is.

Three things printed here appear nowhere else — not in the `Report`, not in the returned object:
the **`EigTarget`** the run is actually given, the **coordinate system** its tolerances live in,
and the **projected cost** before it is paid.

## Warnings

`log_prelanczos` is the only place that knows both the request and the resolved geometry, so it
owns the one warning that needs both: a passband wider than `Defaults.polfed.passband_warn_ratio ×`
the requested window, which means the run will spend its budget converging pairs nobody asked for.
