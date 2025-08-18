module Lanczos2

using LinearAlgebra, CUDA, CUDA.CUSPARSE
using Printf, PrettyTables


# include("EigSorter.jl")
include("ProcessingUnit.jl")
include("ReOrthTechnics/ReOrthTechnics.jl")
include("OrthonormalBasis/OrthonormalBasis.jl")

include("LanczosIterator.jl")
include("Factorization/Factorization.jl")

include("Convergence/ConvergenceInfo.jl")
include("LanczosAlgorithm.jl")



function lanczosmethod(f!::Function, 
                        x₀::AbstractVecOrMat, 
                        howmany::Int; 
                        which::Symbol                       = :smallest,
                        rot::ReOrthTechnique                = FullRO(), 
                        reorth::ReOrthogonalizer            = MatrixGramSchmidt(), 
                        basistype::Type{<:OrthonormalBasis} = MatrixBasis, 
                        maxdim::Int                         = 10howmany, 
                        tol::Real                           = 1e-14, 
                        eigentol::Real                      = 1e-8, 
                        mapvals::Function                   = f!
    ) 


    if isa(reorth, MatrixGramSchmidt) && isa(basistype, Type{VectorBasis}) 
        @warn ("MatrixGramSchmidt reorthogonalization method, not compatible with VectorBasis. Changing to to ClassicalGramSchmidt.")
        reorth = ClassicalGramSchmidt()
    end

    sorter = EigSorter(which)
    pu = isa(x₀, CuArray) ? GPU() : CPU() #determine wether to use GPU or not

    s = size(x₀, 2)
    maxiter = ceil(Int, maxdim/s) # ensure that maxdim is devideble with s
    maxdim = maxiter * s

    krylovbasis = createbasis(maxdim, x₀, rot, basistype, pu)
 
    # println("--------------------")
    # println(typeof(krylovbasis))
    # println(typeof(krylovbasis.fullbasis))
    # println(typeof(krylovbasis.fullbasis.basis))
    # println("--------------------")


    valsconverged, vecsconverged, convergenceinfo_out = lanczosalgorithm(f!, x₀, howmany, maxdim, maxiter, s, pu, krylovbasis, rot, reorth, sorter; tol=tol, eigentol=eigentol, mapvals=mapvals)

    return valsconverged, vecsconverged, convergenceinfo_out
end



export FullRO, PartialRO, ReOrthTechnique
export ClassicalGramSchmidt, MatrixGramSchmidt, ReOrthogonalizer
export MatrixBasis, VectorBasis
export GPU, CPU
export lanczosmethod, lanczosalgorithm, lanczosalgorithm!
export ConvergenceInfoOut
export EigSorter
export display_convergenceinfo, print_convergenceinfo

end
