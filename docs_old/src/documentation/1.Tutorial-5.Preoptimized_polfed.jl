#
# # [Pre-Optimized polfed (POLFED for smart boys)](@id preoptimized_polfed)
# In previous sections we have reduces memory acces as much as possible, in order ot obtain the most performant polfed. With all of these optimizaitons one can get total factor of 4 speedup compared to naive implementation. As u notice most of the opti ization steps were quite general and can be applied to wide range of Hamiltonians (e.g. `XXZ`, `J1-J2`, `Heisenberg`, `Ising`, `QREM` model and many others). That is why we genralized that approach for you, to make your life easier! 

# In the matrix entrypoint of [`polfed`](@ref) you can enable automatic mapping
# optimization via `MappingConfig(optimize_mapping=true)`. This will build an optimized
# mapping, rescaled mapping, and optimized Clenshaw kernels when possible.

using Polfed #hide
include("XXZ.jl")  #hide
L = 14; Nup = L÷2; Δ = 1.0
mat = construct_XXZ_matrix(L, Δ, Nup)
target = 0.0; howmany = 100
v0 = rand(size(mat,1)); v0 ./= norm(v0)
mapping = MappingConfig(optimize_mapping=true)
vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true, mapping=mapping)
display_report(report)

# As you can see from the report, the optimized mapping was used, leading to significant speedup compared to the naive implementation.


# !!! warning "When to use optimized mapping?"
#     Mapping is constructed in a way that it reads out hte different offdiagonal values of the and performs a mapping for each of the values. Idealy all of the offdiagonals would be of the same value, and the speedup would be the best possible. However, if there are multiple offdiagonal values, the mapping will still be constructed, but the speedup might not be as significant good as one would expect. 


# Obviously this is not allways the best possible optimization, for example, in `mixed field ising` model, one can very easily calculate the state that one specific state (`row`) is mapped to. Further reducing the memory acces.
