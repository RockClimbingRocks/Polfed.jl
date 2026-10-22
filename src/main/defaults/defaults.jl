# Defaults registry. One module `Defaults` holding two kinds of submodule:
#
#   per ENTRYPOINT   Defaults.polfed.tol, Defaults.patched_polfed.…, …
#   cross-cutting    Defaults.filters.delta_up.cutoff, Defaults.filters.edge_tol, …
#
# Filters get their own sibling module rather than living under `polfed`, because every
# entrypoint resolves filters. Everything is referenced qualified, so the names never leak into
# `module UniPolfed`.
#
# INCLUDE ORDER — this file is included AFTER `logic/` (see main.jl), because
# `filters/filters.jl` names filter TYPES. That is safe only because every `Defaults.*` reference
# in `logic/` sits in a keyword-argument default, which Julia evaluates at CALL time. Do not
# introduce a top-level `const x = Defaults.…` anywhere in `logic/` — it would load before this.
#
# `Numerics` keeps its own `LanczosDefaults`: it is POLFED-agnostic and must not depend on main/.

module Defaults

include("polfed_defaults.jl")          # module polfed
include("patched_polfed_defaults.jl")  # module patched_polfed
include("full_polfed_defaults.jl")     # module full_polfed
include("double_polfed_defaults.jl")   # module double_polfed (aliases polfed's shared knobs)
include("filters/filters.jl")          # module filters (per-filter knobs + shared geometry)
end # module Defaults
