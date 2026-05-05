# # [Quantum Sun (QSun)](@id qsun_model)
#
# The Quantum Sun model is a compact many-body model for avalanche-driven
# ergodicity restoration. In Polfed it is useful both physically and
# numerically: physically because it separates a small ergodic grain from a set
# of outer spins with decaying couplings, and numerically because it produces
# sparse disordered Hamiltonians that are natural inputs for
# [`polfed`](@ref Polfed.polfed).
#
# The main constructor is [`qsun_hamiltonian`](@ref Polfed.QSun.qsun_hamiltonian):
#
# ```julia
# using Polfed.QSun: qsun_hamiltonian
#
# H = qsun_hamiltonian(L_loc, L_grain, g0, α; kwargs...)
# ```
#
# ## Model Definition
#
# QSun supports two closely related conventions:
# - the default full-Hilbert-space model, which does not enforce U(1)
#   conservation,
# - the U(1)-symmetric model, which works in a fixed total-``S_z`` sector.
# - both constructions generalize directly to higher spins such as `S = 1`,
#   `S = 3/2`, and so on.
#
# Both are built through the same [`qsun_hamiltonian`](@ref Polfed.QSun.qsun_hamiltonian)
# interface.
#
# ### Conventional Quantum Sun Model
#
# In the default construction the Hamiltonian has the form
#
# ```math
# \hat H =
# \hat H_{\mathrm{grain}}
# + g_0 \sum_{j=1}^{L_{\mathrm{loc}}}
# \alpha^{u_j}\, \hat S^x_{n(j)} \hat S^x_j
# + \sum_{j=1}^{L_{\mathrm{loc}}} h_j\, \hat S^z_j .
# ```
#
# Here the first ``L_{\mathrm{grain}}`` sites belong to the ergodic grain, while
# the remaining ``L_{\mathrm{loc}}`` sites are the outer localized spins. In the
# implementation:
# - ``h_j`` are drawn uniformly from ``[h_z - w,\; h_z + w]``.
# - The neighbor map ``n(j)`` picks which grain site couples to the outer spin
#   ``j``.
# - The coupling profile is
#
# ```math
# \alpha^{u_1} = 1, \qquad
# \alpha^{u_j}, \qquad
# u_j = (j-1) + \eta_j, \qquad
# \eta_j \in [-\zeta,\zeta]
# \quad (j \ge 2).
# ```
#
# The parameter ``\zeta`` therefore introduces randomness into the effective
# distance exponent, while ``\alpha`` controls how fast the couplings decay away
# from the grain.
#
# The grain term is constructed as a GOE random matrix:
#
# ```math
# \hat H_{\mathrm{grain}} =
# \frac{\gamma}{\sqrt{d_{\mathrm{grain}} + 1}}\, G,
# ```
#
# where ``G`` is a real symmetric GOE-like matrix and
# ``d_{\mathrm{grain}} = (2S+1)^{L_{\mathrm{grain}}}`` is the grain Hilbert-space
# dimension.
#
# The role of ``\gamma`` is important:
# - ``\gamma`` sets the overall energy scale of the grain block.
# - Larger ``\gamma`` makes the internal grain dynamics stronger relative to the
#   outer-spin disorder and the grain-to-ray couplings.
# - Smaller ``\gamma`` weakens the grain and makes the competition with disorder
#   more pronounced.
#
# In other words, ``\gamma`` does not change the structure of the model, but it
# does change how strongly the ergodic grain acts as a bath for the outer spins.
#
# #### Parameters in the Conventional Model
#
# - `L_loc`: number of outer localized spins.
# - `L_grain`: number of spins inside the ergodic grain.
# - `g0`: overall prefactor multiplying all grain-to-ray couplings.
# - `α`: decay parameter for the long-range couplings. For `0 < α < 1`, the
#   couplings decay exponentially with effective distance; smaller `α` gives
#   faster decay.
# - `γ`: overall scale of the random grain Hamiltonian.
# - `w`: half-width of the uniform disorder window for the local fields. The
#   fields are sampled from `[hz - w, hz + w]`.
# - `hz`: center of the disorder window for the local fields.
# - `ζ`: randomness in the effective distance exponent `u_j`. It broadens the
#   coupling profile around the deterministic value `j - 1`.
# - `S`: on-site spin. It can be integer or half-integer, and the local Hilbert
#   space dimension is `2S + 1`.
# - `rng`: random-number generator controlling disorder, couplings, and the
#   grain matrix.
# - `use_sparse`: if `true`, return a sparse matrix; if `false`, return a dense
#   matrix.
#
# The full Hilbert-space dimension is
#
# ```math
# (2S + 1)^{L_{\mathrm{grain}} + L_{\mathrm{loc}}}.
# ```
#
# ### U(1) Symmetric Quantum Sun Model
#
# The U(1)-symmetric constructor keeps only states in a fixed total-``S_z``
# sector and replaces the grain-to-ray coupling by the U(1)-preserving
# combination
#
# ```math
# \hat S^x_{n(j)} \hat S^x_j + \hat S^y_{n(j)} \hat S^y_j
# =
# \frac{1}{2}
# \left(
# \hat S^+_{n(j)} \hat S^-_j + \hat S^-_{n(j)} \hat S^+_j
# \right).
# ```
#
# The Hamiltonian then reads
#
# ```math
# \hat H_{U(1)} =
# \hat H_{\mathrm{grain}}^{(S_z)}
# + g_0 \sum_{j=1}^{L_{\mathrm{loc}}}
# \alpha^{u_j}\,
# \left(
# \hat S^x_{n(j)} \hat S^x_j + \hat S^y_{n(j)} \hat S^y_j
# \right)
# + \sum_{j=1}^{L_{\mathrm{loc}}} h_j\, \hat S^z_j .
# ```
#
# In this construction, the grain matrix is first organized in a block-diagonal
# way with respect to magnetization sectors inside the grain subspace. The final
# reduced Hamiltonian is then obtained by keeping only matrix elements that
# conserve the total magnetization of the full system, not by fixing the grain
# magnetization independently of the outer spins.
#
# The reduced grain block is therefore obtained by projecting the random grain
# matrix into the chosen total-``S_z`` sector and then rescaling it so that `γ`
# still sets its overall strength:
#
# ```math
# \hat H_{\mathrm{grain}}^{(S_z)}
# =
# \gamma\,
# \frac{R_{\mathrm{cons}}}
# {\sqrt{\mathrm{tr}(R_{\mathrm{cons}}^2)/d_{\mathrm{red}}}} .
# ```
#
# This means that `γ` plays the same conceptual role in both conventions: it is
# the scale of the grain Hamiltonian. The difference is only that in the
# U(1)-symmetric case the grain matrix is first projected into the selected
# sector before being normalized.
#
# #### Additional Parameters in the U(1) Symmetric Model
#
# The parameters `L_loc`, `L_grain`, `g0`, `α`, `γ`, `w`, `hz`, `ζ`, `S`,
# `rng`, and `use_sparse` keep the same meaning as in the conventional model.
# The new ingredients are:
# - `use_U1=true`: tells [`qsun_hamiltonian`](@ref Polfed.QSun.qsun_hamiltonian)
#   to build the U(1)-symmetric model instead of the conventional one.
# - `S_z`: total magnetization sector kept in the reduced Hilbert space.
#
# The chosen sector must satisfy
#
# ```math
# |S_z| \le (L_{\mathrm{grain}} + L_{\mathrm{loc}})S,
# ```
#
# and
#
# ```math
# S_z + (L_{\mathrm{grain}} + L_{\mathrm{loc}})S \in \mathbb{Z}.
# ```
#
# For spin-1/2 and an even number of sites, `S_z = 0.0` is the central sector
# and is usually the most natural first choice.
#
# ## Basic Example
#
# The most direct way to work with QSun is to construct the Hamiltonian with
# [`qsun_hamiltonian`](@ref Polfed.QSun.qsun_hamiltonian) and then pass it into
# [`polfed`](@ref Polfed.polfed).
#
# ### Full-Hilbert-Space Example
#
# ```julia
# using Polfed
# using Polfed.QSun: qsun_hamiltonian
# using LinearAlgebra
# using Random
#
# # random number generator
# rng = MersenneTwister(1234)
#
# # setting model parameters
# L_loc = 8
# L_grain = 2
# g0 = 1.0
# α = 0.55
#
# # constructing the Hamiltonian
# H = qsun_hamiltonian(
#     L_loc,
#     L_grain,
#     g0,
#     α;
#     S=0.5,
#     γ=1.0,
#     w=0.5,
#     hz=1.0,
#     ζ=0.2,
#     rng=rng,
#     use_sparse=true,
# )
#
# # generating a random initial state
# x0 = rand(rng, size(H, 1))
# x0 ./= norm(x0)
#
# # setting POLFED parameters:
# # number of states and position in the spectrum
# howmany = 40
# target = :middle
#
# vals, vecs = polfed(H, x0, howmany, target)
# ```
#
# In this example:
# - `L_grain = 2` builds a small ergodic grain.
# - `L_loc = 8` adds eight outer spins coupled to that grain.
# - `γ = 1.0` sets the strength of the random grain block.
# - `w = 0.5` and `hz = 1.0` define the disorder interval for the local
#   ``S^z`` fields.
# - `ζ = 0.2` makes the effective coupling distances slightly random.
# - `use_sparse=true` returns a sparse matrix, which is typically the right
#   choice for POLFED workflows.
#
# ### U(1)-Symmetric Example
#
# ```julia
# using Random
#
# # random number generator
# rng = MersenneTwister(1234)
#
# # constructing the Hamiltonian in a fixed total-S_z sector
# H_u1 = qsun_hamiltonian(
#     L_loc,
#     L_grain,
#     g0,
#     α;
#     S=0.5,
#     γ=1.0,
#     w=0.5,
#     hz=1.0,
#     ζ=0.2,
#     rng=rng,
#     use_U1=true,
#     S_z=0.0,
#     use_sparse=true,
# )
#
# # checking the reduced Hilbert-space size
# println(size(H_u1))
# ```
#
# This call uses the same physical parameters but restricts the Hamiltonian to a
# fixed total-``S_z`` sector. For larger systems this can reduce the matrix size
# substantially, so it is often the preferred choice when the symmetry is
# relevant for the problem you want to study.
#
# ## Why Quantum Sun?
#
# The Quantum Sun model is a useful benchmark for several reasons:
# - It is a clean toy model of the avalanche picture of ergodicity restoration:
#   a small ergodic grain is coupled to a set of otherwise localized spins.
# - The parameter `α` gives direct control over how quickly couplings decay away
#   from the grain, which makes it a natural tuning parameter for studying
#   ergodicity breaking.
# - The model is easy to vary across disorder realizations and parameter
#   regimes, while still remaining rich enough to display nontrivial many-body
#   behavior.
# - It is particularly well matched to [`polfed`](@ref Polfed.polfed), because
#   one usually wants interior or near-middle-spectrum eigenpairs of fairly
#   large sparse Hamiltonians to study ergodicity-breaking transitions at
#   infinite temperature.
# - The U(1)-symmetric version provides a second, smaller representation of the
#   same physical idea, which is very helpful when exploring larger systems or
#   specific symmetry sectors.
#
# The Quantum Sun model also serves as a natural playground for universal
# features near a many-body ergodicity-breaking transition (EBT). In
# particular, the way eigenstate thermalization breaks down as the transition is
# approached provides a useful framework for detecting an upcoming EBT; see for
# example [Fading ergodicity meets maximal chaos (PRB 111, 184203)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.184203)
# and [Fading ergodicity (PRB 110, 134206)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.134206).
#
# The non-ergodic regime is equally informative: it allows one to quantify
# localization properties in many-body Hilbert space and to track the emergence
# of area-law or sub-volume-law entanglement structures. Recent variants of the
# model also show that nonstandard combinations of spectral statistics and
# eigenstate structure can appear in generalized Quantum Sun settings; see
# [Many-body mobility edge in quantum sun models (PRB 109, L180201)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.L180201)
# and [Unconventional Thermalization of a Localized Chain Interacting with an Ergodic Bath (arXiv:2507.18286)](https://arxiv.org/abs/2507.18286).
#
# In short, QSun is a good model when you want a physically meaningful,
# disorder-driven many-body problem that is still straightforward to generate
# and vary across many realizations.
#
# ## Related Literature
#
# - [Fading ergodicity (PRB 110, 134206)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.134206)
# - [Fading ergodicity meets maximal chaos (PRB 111, 184203)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.184203)
# - [Ergodicity Breaking Transition in Zero Dimensions (PRL 129, 060602)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.060602)
# - [Similarity between quantum sun and ultrametric random matrix model (PRR 6, 023030)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.023030)
# - [Many-body mobility edge in quantum sun models (PRB 109, L180201)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.L180201)
# - [Unconventional Thermalization of a Localized Chain Interacting with an Ergodic Bath (arXiv:2507.18286)](https://arxiv.org/abs/2507.18286)
