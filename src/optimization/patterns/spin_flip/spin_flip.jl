# The spin_flip pattern: the neighbour is COMPUTED, never read.
#
#   H = diag(ε) + Σ_k v_k · X^{mask_k}          on the full space, D = 2^L
#
# A mask of 1 bit is σˣ_j (transverse field, QREM); 2 bits is σˣ_a σˣ_b (the Quantum Sun
# dot-spin coupling); n bits is any product of X operators. Because `xor(i-1, mask)` gives the
# neighbour outright, this pattern stores NO index table — which is both why it is fast and why
# it reaches larger L at fixed memory than any read-out kernel can.
#
# Everything here assumes the full 2^L space. In a fixed-Sz sector the flipped bitstring still
# has to be mapped back to a sector index, which is a lookup — a different pattern, not this one.

include("FlipStructure.jl")     # the term list + normalisation + rescale + derivation from H
include("kernels/kernels.jl")   # the kernels themselves
include("build_kernel.jl")      # (variant, structure, bs) → the closure
include("validate_kernel.jl")   # may these kernels run on this structure, and at this bs?
