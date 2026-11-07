# The universal pattern: value-grouped kernels that read each group's own per-row extents, so they
# run on any structure `classify` can produce — several distinct off-diagonal values, rows of
# differing length, or both.

include("kernels/kernels.jl")   # the kernels themselves
include("build_kernel.jl")      # (variant, structure, bs, params) → the closure
include("validate_kernel.jl")   # may these kernels run on this structure?
