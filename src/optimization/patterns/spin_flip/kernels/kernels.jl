# The kernels this pattern provides. One include per kernel file; each defines its own
# `_build_<variant>` used by ../build_kernel.jl.

include("spin_flip_rows_kernel.jl")
include("spin_flip_packed_kernel.jl")
