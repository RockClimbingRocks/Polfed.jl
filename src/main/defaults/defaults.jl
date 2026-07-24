# Defaults registry. One module `Defaults` with a per-function submodule each (`Defaults.polfed`,
# `Defaults.patched_polfed`, …), so each entrypoint's default knobs (tol, N, kernel, …) are grouped
# and referenced qualified — e.g. `Defaults.polfed.tol`. Included first: the stage configs
# (DoSConfig / MappingConfig) and the pipelines read these defaults.

module Defaults

include("polfed_defaults.jl")           # module polfed
include("patched_polfed_defaults.jl")   # module patched_polfed
include("full_polfed_defaults.jl")     # module full_polfed
end # module Defaults
