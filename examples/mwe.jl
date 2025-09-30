# # Minimal Working Example of polfed
#-
#md # !!! Tip:
#md #     This example is also available as a Jupyter notebook:
#md #     bla bla bla (here it was a reference to some notebook )
#-
#-
# ## Installation
# To install Polfed, you can use the Julia package manager. In the Julia REPL, type
# import Pkg
# Pkg.add("Polfed")
# Polfed.jl is a pure Julia package; no dependencies (aside from the Julia standard library) are required.
# ## Usage
# Once Polfed is installed, we can include the Polfed package in our code:
using Polfed, Random, SparseArrays, LinearAlgebra
# To solve eigenvalue problem 
# ```math
#   H \vec{u} = E \vec{u}
# ```
# at the targeted part of the spectrum $\lambda$ we will consider the paradigmatic XXZ model
# ```math
#   H_{XXZ} = \sum_{i=1}^{L-1} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + \Delta S_i^z S_{i+1}^z)
# ```
# with $S_i^{x,y,z}$ being the spin-$1/2$ operators at site $i$, $\Delta$ the anisotropy parameter. The model is defined on a chain of length $L$ with periodic boundary conditions. The total magnetization $S^z_{tot} = \sum_i S_i^z$ is conserved, and we will work in the zero magnetization sector.
# Let us construct the function for constructing the matrix representation of the Hamiltonian
function construct_xxz(L::Int, Δ::Real; Nup::Int=L÷2)
    basis = [b for b in 0:2^L-1 if count_ones(b) == Nup] # generate basis
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))  # state index map
    rows, cols, vals = Int[], Int[], Float64[]
    for (col, state) in enumerate(basis)
        for i in 1:L
            j = i % L + 1  # PBC
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1
            SzSz = (0.5 - si) * (0.5 - sj) 
            push!(rows, col); push!(cols, col); push!(vals, Δ * SzSz)
            if si != sj
                flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                if haskey(bmap, flipped) 
                    push!(rows, bmap[flipped]); push!(cols, col); push!(vals, 0.5)
                end
            end
        end
    end
    return sparse(rows, cols, vals, dim, dim)
end
# We can now construct the Hamiltonian matrix for a chain of length `L=18` with anisotropy parameter `Δ=1.0` 
L = 14; Δ = 1.0
mat = construct_xxz(L, Δ)
# and prepare a **normalized** random initial vector `v0` in the Hilbert space of the model, as orthogonal as possible to the targeted eigenvectors,  target energy $\lambda$, and number of desired eigenpairs `howmany`
Random.seed!(1234) # for reproducibility
v0 = rand(size(mat,1)); v0 ./= norm(v0)
λ = 0.0; 
howmany = 100
# We can now call the [`polfed`](@ref) function to compute the eigenvalues and eigenvectors
vals, vecs = @time Polfed.polfed(mat, v0, howmany, λ)
display(vals)
# The computed eigenvalues are displayed above. We can also check that the eigenpairs are correct by computing the residual norms $||H \vec{u} - E \vec{u}||$ for each eigenpair $(E, \vec{u})$, which we also verify inside the algorithm.
# To better understand what happens during the run, one can enable the reporting option by setting the corresponding keyword argument to \texttt{true}. 
vals, vecs, report = @time Polfed.polfed(mat, v0, howmany, λ; produce_report=true)
Polfed.display_report(report)
# The reporting system is flexible: you can include or exclude parts of the report (e.g., convergence details, benchmarks), see [`display_report`](@ref) for more details.
# ```@docs
# display_report
# ```

