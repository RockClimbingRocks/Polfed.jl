

function lanczos_method(
    iterator::LanczosIterator{E,T,ROT}, 
    convergenceinfo::ConvergenceInfo, 
    basistype::Type{<:OrthonormalBasis}, 
    pu::ProcessingUnit
) where {E<:Real,T<:AbstractVecOrMat,ROT<:ReOrthTechnique}
    hilbertspacedim = size(iterator.x0, 1)
    howmany = convergenceinfo.howmany

    vals = pu.Vector{E}(undef, howmany)
    vecs = pu.Matrix{E}(undef, (hilbertspacedim, howmany))


    factorization_report = lanczos_method!(vals, vecs, iterator, convergenceinfo, basistype, pu)

    converged = min(convergenceinfo.converged, convergenceinfo.eigenconverged)
    vals_   = view(vals, 1:converged)
    idxs    = sortvals(vals_, EigSorter(:smallest))
    
    vals_converged_ordered = pu.Vector(view(vals,   idxs))
    vecs_converged_ordered = pu.Matrix(view(vecs, :,idxs))

    return vals_converged_ordered, vecs_converged_ordered, factorization_report
end



function lanczos_method!(
    vals::AbstractVector, 
    vecs::AbstractMatrix, 
    iterator::LanczosIterator, 
    convergenceinfo::ConvergenceInfo, 
    basistype::Type{<:OrthonormalBasis}, 
    pu::ProcessingUnit,
)
    # Initialize the Krylov basis
    krylovbasis = createbasis(convergenceinfo.maxdim, iterator.x0, basistype, pu)


    # println("Starting Lanczos method with $(convergenceinfo.howmany) eigenvalues and vectors...")

    # println("LanczosIterator fields and values:")
    # for field in fieldnames(typeof(iterator))
    #     println("  $(field): ", getfield(iterator, field))
    # end

    # println("ConvergenceInfo fields and values:")
    # for field in fieldnames(typeof(convergenceinfo))
    #     println("  $(field): ", getfield(convergenceinfo, field))
    # end
    # Run the Lanczos algorithm
    lanczos_algorithm!(vals, vecs, iterator, convergenceinfo, krylovbasis, pu)
end

