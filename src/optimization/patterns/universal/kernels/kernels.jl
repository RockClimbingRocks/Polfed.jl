# The kernels this pattern provides. One include per kernel file; each defines its own
# `_build_<variant>` used by ../build_kernel.jl.

include("universal_row_read_kernel.jl")
include("universal_row_bucketed_kernel.jl")
