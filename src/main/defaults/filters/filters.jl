# Filter defaults — the cross-cutting slice of the registry. Filters are NOT owned by one
# entrypoint (`polfed`, `patched_polfed`, `full_polfed` and `double_polfed` all resolve filters),
# so these live as a SIBLING of the per-entrypoint modules rather than nested inside one:
#
#   Defaults.polfed.tol                 per entrypoint
#   Defaults.filters.delta_up.cutoff    cross-cutting  ← here
#
# One submodule per filter, named exactly like its folder under `logic/filter/filter_types/`.
#
# NOTE — this file is why `defaults/` is included AFTER `logic/` (see main.jl): the resolution
# chains below name filter TYPES, which do not exist until the filter files have loaded.

"""
    UniPolfed.Defaults.filters

Filter defaults, one submodule per filter (`Defaults.filters.delta_up.cutoff`, …), plus the
filter-agnostic geometry knobs shared by the resolution machinery.

Cross-cutting by design: every entrypoint resolves filters, so these are *not* nested under
[`Defaults.polfed`](@ref). Values are referenced qualified so `cutoff`/`order`/… never collide
between filters.
"""
module filters

import ...DeltaUpFilter
import ...DeltaDownFilter
import ...DeltaUpSmoothFilter
import ...EdgeFilter

include("delta_up.jl")
include("delta_down.jl")
include("delta_up_smooth.jl")
include("double_delta.jl")
include("edge.jl")

# ── shared geometry ───────────────────────────────────────────────────────────

"""
    Defaults.filters.edge_tol::Float64 = 1e-9

Tolerance for *"this window's bound sits ON a spectrum edge"*, shared by [`edge_side`](@ref)
and the [`EdgeFilter`](@ref) build.

This is a **containment** test, not a distance one: it asks whether the requested window
reaches the spectrum extreme (`±1` in rescaled coordinates), because a window that contains the
extreme needs the monotone `EdgeFilter` while every other window — including a near-edge one
like `[-0.99, -0.7]` — belongs to the delta family. The pipelines clamp overhanging windows to
exactly `±1`, so only rounding-scale slack from the rescale `(E - b)/a` needs absorbing; the
nearest genuine interior case sits `~1e-2` away, seven orders of magnitude clear.

Distinct from `Defaults.patched_polfed.edge_distance` (`0.2`), which answers a different
question: given that an edge IS being peeled, how wide should the peel be.
"""
const edge_tol = 1e-9

# ── resolution chains ─────────────────────────────────────────────────────────

"""
    Defaults.filters.INTERIOR_FILTER_CHAIN

Filter types tried, in order, for an interior window — first valid wins (see
[`resolve_filter`](@ref)). The two sharp deltas come first because they are the cheapest;
[`DeltaUpSmoothFilter`](@ref) is the fallback for windows they cannot serve cleanly.
"""
const INTERIOR_FILTER_CHAIN = (DeltaUpFilter, DeltaDownFilter, DeltaUpSmoothFilter)

"""
    Defaults.filters.EDGE_FILTER_CHAIN

Filter types tried, in order, for a window that touches a spectrum edge (`±1`).
"""
const EDGE_FILTER_CHAIN = (EdgeFilter,)

end # module filters
