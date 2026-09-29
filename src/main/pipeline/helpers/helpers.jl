
include("filtered_lanczos.jl")   # the filtered Lanczos call every pipeline makes
include("intervalize_interval.jl")
include("result_buffer.jl")
include("resolve_edges.jl")
include("resolve_interior.jl")
include("double_helpers.jl")   # shared machinery for the two double_polfed pipelines
