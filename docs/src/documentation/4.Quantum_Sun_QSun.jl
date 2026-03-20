# # [Quantum Sun (QSun)](@id qsun_model)
#
# The Quantum Sun model is a paradigmatic toy model for avalanche-driven
# ergodicity restoration in many-body systems. It is widely used because it is
# simple enough for controlled analysis, but still captures a sharp
# ergodicity-breaking transition.
#
# In this documentation, QSun is the default beginner model because it is easy
# to construct and run while still being physically meaningful.
#
# ## Model Definition
#
# Following Eq. (7) of
# [Kliczkowski et al. (2024)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.134206), the Hamiltonian
# can be written as
#
# ```math
# \hat H = \hat H_{\mathrm{dot}} + \sum_{j=1}^{L} \alpha^{u_j}\,\hat S^x_{n(j)}\hat S^x_j
# + \sum_{j=1}^{L} h_j\,\hat S^z_j.
# ```
#
# Interpretation:
# - ``H_{\mathrm{dot}}``: a small ergodic quantum dot (the thermal seed),
# - ``L`` outer spins ("rays") with local fields ``h_j``,
# - exponentially decaying couplings ``\alpha^{u_j}`` between rays and dot spins.
#
# The control parameter ``\alpha`` tunes the system across the ergodicity-breaking
# transition. In avalanche-theory treatments, the critical value is predicted as
#
# ```math
# \alpha_c = \frac{1}{\sqrt{2}}.
# ```
#
# ## Why This Model Is Important
#
# - It is a clean realization of "ergodic grain + localized environment"
#   avalanche physics.
# - Key ergodicity diagnostics (e.g. Thouless scales) admit controlled
#   analytical estimates.
# - It is a strong benchmark for comparing numerical methods near criticality.
#
# ## Physical Picture: Why a Small Ergodic Grain Can Delocalize the Rest
#
# Let the effective coupling of spin ``j`` to the grain be
# ``J_j \sim \alpha^{u_j}``.
# If the current grain size is ``M``, a rough many-body level-spacing scale is
# ``\delta_M \sim 2^{-M}``.
#
# A new spin gets absorbed when hybridization is strong enough
# (``J_j \gtrsim \delta_M``). Once absorbed, the grain grows
# (``M \to M+1``) and level spacing gets
# smaller, making further absorption easier. This feedback is the avalanche
# mechanism.
#
# - For ``\alpha > \alpha_c``, the avalanche can continue and the system becomes
#   ergodic.
# - For ``\alpha < \alpha_c``, couplings decay too quickly and avalanche growth
#   stalls, leaving non-ergodic behavior.
#
# ## How It Is Used in This Documentation
#
# Typical constructor used in beginner tutorials:
#
# ```julia
# using Polfed
# using Polfed.QSun: quantum_sun_hamiltonian
#
# mat = quantum_sun_hamiltonian(12, 2; sparse=true)
# ```
#
# Then call [`polfed`](@ref Polfed.polfed) with either:
# - a normalized vector (`x0::AbstractVector`) for Lanczos,
# - or a normalized block (`x0::AbstractMatrix`) for Block Lanczos.
#
# ## Related Literature
#
# - [Fading ergodicity (PRB 110, 134206)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.134206)
# - [Fading ergodicity meets maximal chaos (PRB 111, 184203)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.184203)
# - [Ergodicity Breaking Transition in Zero Dimensions (PRL 129, 060602)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.060602)
# - [Similarity between quantum sun and ultrametric random matrix model (PRR 6, 023030)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.023030)
# - [Many-body mobility edge in quantum sun models (PRB 109, L180201)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.L180201)
# - [How Random Are Ergodic Eigenstates of the Ultrametric Random Matrices and the Quantum Sun Model? (arXiv:2501.19244)](https://arxiv.org/abs/2501.19244)
# - [Many-body localization and Poisson statistics in the Quantum Sun model (arXiv:2506.13511)](https://arxiv.org/abs/2506.13511)
# - [Unconventional thermalization of a localized chain interacting with an ergodic bath: Interacting Anderson and quantum sun model (arXiv:2507.18286)](https://arxiv.org/abs/2507.18286)
