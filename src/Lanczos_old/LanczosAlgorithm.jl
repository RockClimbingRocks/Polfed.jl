

function lanczosalgorithm!( f!::Function, 
                            x₀::AbstractVecOrMat{E}, 
                            howmany::Int, 
                            maxdim::Int, 
                            maxiter::Int,
                            krylovdim::Int,
                            pu::ProcessingUnit,
                            krylovbasis::Basis, 
                            rot::ReOrthTechnique, 
                            reorth::ReOrthogonalizer,
                            vals::AbstractVector,
                            vecs::AbstractMatrix,
                            sorter::EigSorter;
                            tol::Real=1e-14, 
                            eigentol::Real=1e-8, 
                            mapvals::Function=f!) where {E<:Real}

    LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

    # MethodError: no method matching QSystem.Lanczos.ConvergenceInfo(::Int64, ::Int64, ::Int64, ::Float64, ::QSystem.La`nczos.EigSorter)
    # The type `QSystem.Lanczos.ConvergenceInfo` exists, but no method is defined for this combination of argument types` when trying to construct it.
    convergenceinfo = ConvergenceInfo(howmany, krylovdim, maxiter, tol, sorter)
    valsinfo        = EigenvaluesInfo(eigentol, mapvals)

    iterator        = LanczosIterator(f!, x₀, rot, reorth)
    factorization   = lanczositer(iterator, krylovbasis, maxdim, pu)
    convergenceinfo.numiter+=1
    

    times = zeros(6)
    while true
        lanczositer!(iterator, factorization, times)
        times[6] +=  @elapsed getnumberofconvergedvecs!(convergenceinfo, factorization, valsinfo, vecs, vals)

        convergenceinfo.converged == howmany && break
        convergenceinfo.numiter >= maxiter && break
    end

    # if valsinfo.converged < howmany
    #     @info "Number of states with residual smaller then $(convergenceinfo.tol): $(convergenceinfo.converged) (largest residual: $(convergenceinfo.residual))"
    #     @info "Number of eigenpairs with residual norm smaller then $(valsinfo.tol): $(valsinfo.converged) (largest residual: $(valsinfo.eigenresidual))"
    # end    
    # @info "Number of iterations needed: $(convergenceinfo.numiter)   (out of $(convergenceinfo.maxiter) reserved)"

    
    convergenceinfo_out = ConvergenceInfoOut(convergenceinfo, valsinfo, factorization, times)
    


    display_convergenceinfo(convergenceinfo_out)
    # println("valss-------- converged: $(valsinfo.converged)")
    # display(vals)

    vals_   = view(vals, 1:valsinfo.converged)
    idxs    = sortvals(vals_, EigSorter(:smallest))
    
    valsconverged = view(vals,   idxs)
    vecsconverged = view(vecs, :,idxs); isa(pu,GPU) && (valsconverged = CuVector(valsconverged))

    return valsconverged, vecsconverged, convergenceinfo_out
end





function lanczosalgorithm(f!::Function, 
                          x₀::AbstractVecOrMat{E}, 
                          howmany::Int, 
                          maxdim::Int,
                          maxiter::Int,
                          s::Int,
                          pu::ProcessingUnit,
                          krylovbasis::Basis, 
                          rot::ReOrthTechnique, 
                          reorth::ReOrthogonalizer,
                          sorter::EigSorter;
                          tol::Real=1e-14, 
                          eigentol::Real=1e-8, 
                          mapvals::Function=f!) where {E<:Real}

    hilbertspacedim = size(x₀,1)

    vals = pu.vec{E}(undef, howmany)
    vecs = pu.mat{E}(undef, hilbertspacedim, howmany)

    valsconverged, vecsconverged, convergenceinfo_out = lanczosalgorithm!(f!, x₀, howmany, maxdim, maxiter, s, pu, krylovbasis, rot, reorth, vals, vecs, sorter; tol=tol, eigentol=eigentol, mapvals=mapvals)


    return valsconverged, vecsconverged, convergenceinfo_out
end
