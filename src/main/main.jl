# main/ tier aggregator. Includes the POLFED stage slices in dependency order, then the pipeline
# that sequences them, then the public entrypoints that wrap the pipeline. No logic lives here —
# each concern is its own folder with its own aggregator (like every other slice in the tree).

include("sys_info.jl")            # SysInfo — problem shape/type/device (carried on the plan)

# ── logic ──────────────
include("logic/logic.jl")

# Defaults AFTER logic: `Defaults.filters` names filter TYPES, so they must already exist.
# Safe because every `Defaults.*` use in logic/ is a keyword-argument default (evaluated at CALL
# time). Never write a top-level `const x = Defaults.…` inside logic/.
include("defaults/defaults.jl")   # Defaults: per-entrypoint registries + Defaults.filters

# ── pipeline + entrypoints ─────────────────────────────────────────────────────
include("pipeline/pipelines.jl")      # per-function pipelines (run_polfed_pipeline, …)
include("entrypoint/entrypoint.jl")   # polfed(f!, …) / polfed(H, …) + variants, thin over the pipelines
