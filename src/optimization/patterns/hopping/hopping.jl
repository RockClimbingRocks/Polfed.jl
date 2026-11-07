# The hopping pattern: the neighbour is COMPUTED, and axis 1 is not even loaded.
#
#   H = diag(ε) + t · Σ_{k=1}^{D} (T_k⁺ + T_k⁻)     on a periodic D-dimensional lattice
#
# The Anderson model: on-site disorder plus uniform nearest-neighbour hopping, in any dimension.
# The neighbour is a site index ± a stride with one wrap test per axis, so this pattern stores NO
# index table — 168 MB of neighbour columns at 128³ that the shape tuple already implies.
#
# Both kernels SLIDE along axis 1: each site is loaded once and used as the next step's centre
# and then as its left neighbour, so the 2D+1-point stencil costs 2D-1 input streams. That is the
# measured difference between this pattern and a generic stencil — 1.5x to 8.6x on the Zen 4
# sweep in notebooks/benchmarks/matvec/hop, widest at small block sizes.
#
# `D` is a TYPE parameter throughout. A runtime-dimensional variant of the same arithmetic was
# measured at 3.7-5.7x slower, so the dimension must reach the compiler before the apply loop.
# Up to `MAX_UNROLLED_DIM` the whole stencil is unrolled; above it only the orthogonal sum
# becomes a counted loop.

include("HopStructure.jl")      # the lattice + validation + rescale + derivation from H
include("kernels/kernels.jl")   # the kernels themselves
include("build_kernel.jl")      # (variant, structure, bs) → the closure
include("validate_kernel.jl")   # may these kernels run on this structure, and at this bs?
