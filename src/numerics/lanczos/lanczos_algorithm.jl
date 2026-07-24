"""
    lanczos_algorithm(iterator, criterion, state, krylovbasis, pu) -> (vals, vecs, residuals, report)

Run the core Lanczos iteration loop until the target's stopping criterion is met
or the iteration budget is exhausted, then return the extracted converged
eigenpairs and a [`FactorizationReport`](@ref). The number of eigenpairs is
determined dynamically (value-based targets), so outputs are allocated at extraction.
"""
function lanczos_algorithm(
    iterator::LanczosIterator,
    criterion::ConvergenceCriterion,
    state::ConvergenceState,
    krylovbasis::OrthonormalBasis,
    pu::ProcessingUnit,
)
    # Thread configuration (BLAS/MKL/Julia) is intentionally NOT mutated here: it
    # is a global side effect that can conflict with the caller's strategy and with
    # `f!`'s own parallelization (e.g. POLFED's per-mapping ParallelizationStrategy).
    factorization = lanczositer(iterator, krylovbasis, criterion.maxdim, pu)
    state.numiter += 1
    polfed_log(
        POLFED_DEBUG_LEVEL,
        "Lanczos initialized.",
        iteration=state.numiter,
        krylovdim=factorization.krylovdim,
        blocksize=(factorization isa LanczosFactorization ? 1 : factorization.blocksize),
    )

    result = nothing
    # Open the stage-timing scope for the whole loop: `@stage` annotations here AND inside
    # `lanczositer!` accumulate by name into `timer`, summing over every iteration.
    timer = @stages while true
        iter = state.numiter + 1
        ok = lanczositer!(iterator, factorization, iter)

        # On breakdown, force a final extraction: the invariant subspace's Ritz
        # pairs are exact eigenpairs of the targeted spectrum.
        checked = @stage :conv_check check_convergence!(criterion, state, factorization; force=!ok)
        checked === nothing || (result = checked)

        # DEEP, not DEBUG: this fires on EVERY iteration (hundreds to thousands per run), which
        # would drown the phase-level messages that make level 3 readable.
        polfed_log(
            POLFED_DEEP_DEBUG_LEVEL,
            "Lanczos iteration finished.",
            iteration=state.numiter, krylovdim=factorization.krylovdim,
            converged=state.converged, residual=state.residual,
        )

        ok || break                                  # breakdown: invariant subspace reached, no more Krylov vectors to add
        state.done && break                          # success: the target's full request is converged by both Ritz and eigen gates
        state.numiter >= criterion.maxiter && break  # budget exhausted: stop and return whatever fully converged so far
    end

    if result === nothing
        T = eltype(iterator.x0)
        S = real(T)
        dim = opdim(iterator.layout, iterator.x0)
        emptyvecs = iterator.layout isa RowLayout ? pu.Matrix{T}(undef, 0, dim) : pu.Matrix{T}(undef, dim, 0)
        result = (pu.Vector{S}(undef, 0), emptyvecs, Float64[])
    end
    vals, vecs, residuals = result

    report = FactorizationReport(criterion, state, factorization, iterator, timer)
    return vals, vecs, residuals, report
end
