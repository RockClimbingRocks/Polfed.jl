# Concrete filters — one self-contained folder each. Adding a filter = add a folder + one
# include line here, implement the resolution interface (`_cutoff_values`/`_build_candidate`,
# see filter.jl; fitness lives in the filter's own `validate_filter`), and append the type to a
# resolution chain (`Defaults.filters.INTERIOR_FILTER_CHAIN`/`Defaults.filters.EDGE_FILTER_CHAIN` in resolution/resolve_filter.jl).

include("delta_up/delta_up.jl")
include("delta_down/delta_down.jl")
include("delta_up_smooth/delta_up_smooth.jl")
include("double_delta/double_delta.jl")
include("edge/edge.jl")