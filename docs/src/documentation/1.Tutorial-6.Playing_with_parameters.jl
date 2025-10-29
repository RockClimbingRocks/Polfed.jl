#
# # [Other [`polfed`] functionalities](@id polfed_functionalities)
# Lets get a bit deeper ino the [`polfed`](@ref) functionalities, and see what other parameters can be played with. In previous tutorials we have already seen how to run a basic polfed calculation, now we will explore some of the additional options that can be used, and are ussually not performance related.

using Polfed #hide
include("XXZ.jl")  #hide
L = 14; Nup = L÷2; Δ = 1.0
mat = construct_XXZ_matrix(L, Δ, Nup)
target = 0.0; howmany = 100
v0 = rand(size(mat,1)); v0 ./= norm(v0)
vals, vecs, report = polfed(mat, v0, howmany, target; produce_report=true, optimize_mapping=true)
display_report(report)

# As you can see from the report, the optimized mapping was used, leading to significant speedup compared to the naive implementation. 

# Obviously this is not allways the best possible optimization, for example, in `mixed field ising` model, one can very easily calculate the state that one specific state (`row`) is mapped to. Further reducing the memory acces.

