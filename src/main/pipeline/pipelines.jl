# Pipeline slice. Include after the mapping/DoS/target/filter/transform/report logic.

include("helpers/helpers.jl")

include("pipeline_polfed_target.jl")
include("pipeline_polfed_window.jl")
include("pipeline_patched_polfed_target.jl")
include("pipeline_patched_polfed_window.jl")
include("pipeline_full_polfed.jl")
