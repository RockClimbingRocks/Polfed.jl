# main/ tier aggregator. Includes the POLFED stage slices in dependency order, then the pipeline
# that sequences them, then the public entrypoints that wrap the pipeline. No logic lives here —
# each concern is its own folder with its own aggregator (like every other slice in the tree).

include("defaults/defaults.jl")   # per-entrypoint registries under Defaults
include("sys_info.jl")            # SysInfo — problem shape/type/device (carried on the plan)

# ── logic ──────────────
include("logic/logic.jl")

# ── pipeline + entrypoints ─────────────────────────────────────────────────────
include("pipeline/pipelines.jl")      # per-function pipelines (run_polfed_pipeline, …)
include("entrypoint/entrypoint.jl")   # polfed(f!, …) / polfed(H, …) + variants, thin over the pipelines
