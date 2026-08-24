using UniPolfed, LinearAlgebra, Random
using UniPolfed.Models

L=parse(Int, ARGS[1])
bs = parse(Int, ARGS[2])

# One seeded stream feeds BOTH random inputs (the disorder fields and the Krylov start block),
# so a re-run with the same L/bs reproduces the run exactly. MersenneTwister rather than the
# default RNG: its stream is stable across Julia versions and unaffected by anything else in the
# session drawing numbers. Optional third argument overrides the seed.
const SEED = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20260824
rng = MersenneTwister(SEED)

mat = UniPolfed.Models.xxz_hamiltonian(L, L÷2, 1.0, 1.0, 1.0; boundary = :periodic, rng = rng)
dim = size(mat, 1)
println("dim = $dim, block size = $bs, #blocks = $(div(dim,bs)), seed = $SEED")

x0 = Matrix(lq(rand(rng, bs, dim)).Q)

vals,vecs,report = UniPolfed.polfed(mat,x0,1000,:maxdos; overestimate_iters=1.3)

UniPolfed.display_report(report; convergence_details = true)



vals,vecs,report = UniPolfed.polfed(mat,x0,1000,:maxdos; overestimate_iters=1.5, mc=UniPolfed.MappingConfig(optimize_mapping=true))

UniPolfed.display_report(report; convergence_details = true)
