
function lanczos_algorithm!(
    vals::AbstractVector,
    vecs::AbstractMatrix,
    iterator:: LanczosIterator,
    convergenceinfo::ConvergenceInfo,
    krylovbasis::OrthonormalBasis,
    pu::ProcessingUnit,
)
    LinearAlgebra.BLAS.set_num_threads(Base.Threads.nthreads())

    factorization = lanczositer(iterator, krylovbasis, convergenceinfo.maxdim, pu)
    convergenceinfo.numiter += 1

    walltimes = zeros(6)
    cputimes  = zeros(6)

    while true
        lanczositer!(iterator, factorization, walltimes, cputimes)
        @addtime! walltimes cputimes 6 getnumberofconvergedvecs!(convergenceinfo, factorization, vecs, vals)

        convergenceinfo.converged == convergenceinfo.howmany && break
        convergenceinfo.numiter >= convergenceinfo.maxiter && break
    end

    factorization_report = FactorizationReport(convergenceinfo, factorization, iterator, walltimes, cputimes)
    return factorization_report
end

    