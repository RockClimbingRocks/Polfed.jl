# Entrypoint slice: the public POLFED API — one file per function. Each entrypoint is thin: it
# prepares its inputs and calls that function's pipeline (`run_<name>_pipeline` in pipeline/). New
# variants slot in here as new files (see patched_polfed.jl for the pattern). The operator-building
# helpers the matrix path uses live with the mapping slice (logic/mapping/row_operators.jl).

include("entrypoint_helpers.jl") # shared matrix row-prep for the polfed(H, …) methods
include("polfed.jl")             # polfed(f!, …) and polfed(H, …) — the canonical entrypoints
include("patched_polfed.jl")     # patched target/window operator + matrix entrypoints
include("full_polfed.jl")        # full-spectrum operator + matrix entrypoints
