
"""
    lanczos_algorithm!(vals, vecs, iterator, convergenceinfo, krylovbasis, pu) -> FactorizationReport

Run the core Lanczos iteration loop until convergence or iteration budget is
exhausted.

`vals` and `vecs` are mutable output buffers. Returns a
[`FactorizationReport`](@ref) summarizing the run.
"""
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
    polfed_log(
        POLFED_DEBUG_LEVEL,
        "Lanczos initialized.",
        iteration=convergenceinfo.numiter,
        krylovdim=factorization.krylovdim,
        blocksize=(factorization isa LanczosFactorization ? 1 : factorization.blocksize),
    )

    walltimes = zeros(6)
    cputimes  = zeros(6)

    while true
        iter = convergenceinfo.numiter + 1
        lanczositer!(iterator, factorization, walltimes, cputimes, iter)

        @addtime! walltimes cputimes 6 getnumberofconvergedvecs!(convergenceinfo, factorization, vecs, vals)
        polfed_log(
            POLFED_DEBUG_LEVEL,
            "Lanczos iteration finished.",
            iteration=convergenceinfo.numiter,
            krylovdim=factorization.krylovdim,
            converged=convergenceinfo.converged,
            residual=convergenceinfo.residual,
        )

        convergenceinfo.converged == convergenceinfo.howmany && break
        convergenceinfo.numiter >= convergenceinfo.maxiter && break
    end

    factorization_report = FactorizationReport(convergenceinfo, factorization, iterator, walltimes, cputimes)
    return factorization_report
end

    
