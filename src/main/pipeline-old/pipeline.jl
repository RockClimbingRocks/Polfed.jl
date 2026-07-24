# Pipeline slice: one pipeline per entrypoint function (`run_<name>_pipeline`), sequencing the stage
# slices (mapping/dos/target/interval/filter/transform/report). Included after those slices and
# before the entrypoints that wrap it.



include("helpers/helpers.jl")
include("pipeline_polfed_target.jl")     # polfed (count-based / point target)
include("pipeline_polfed_window.jl")     # polfed (window target)
include("patched_core.jl")               # shared edge-peel + interior-tile core (patched variants)
include("pipeline_patched_polfed_window.jl")
include("pipeline_patched_polfed_target.jl")
include("pipeline_full_polfed.jl")

