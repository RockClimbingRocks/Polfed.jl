# The kernels this pattern provides. One include per kernel file; each defines its own
# `_build_hop_<variant>` used by ../build_kernel.jl. The shared file first: both kernels use the
# same dimension cut, the same per-line neighbour bases, and the same work decomposition.

include("hop_kernel_common.jl")
include("hopping_rows_kernel.jl")
include("hopping_packed_kernel.jl")
