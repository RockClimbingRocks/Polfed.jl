"""
    LanczosResult

Result of a [`lanczos`](@ref) run.

# Fields
- `vals`: converged eigenvalues, sorted ascending.
- `vecs`: corresponding eigenvectors (columns).
- `residuals::Vector{Float64}`: per-eigenvalue true residual `‖A·vᵢ − λᵢ·vᵢ‖`,
  aligned with `vals`.
- `converged::Int`: number of converged eigenpairs.
- `report::FactorizationReport`: timing/convergence diagnostics.

For backward compatibility the result iterates as `(vals, vecs, report)`, so
`vals, vecs, report = lanczos(...)` still works; use [`has_converged`](@ref) and
the `residuals` field for per-eigenvalue queries.
"""
struct LanczosResult{V<:AbstractVector, M<:AbstractMatrix}
    vals::V
    vecs::M
    residuals::Vector{Float64}
    converged::Int
    report::FactorizationReport
end

# Backward-compatible destructuring: `vals, vecs, report = result`.
Base.iterate(r::LanczosResult) = (r.vals, Val(:vecs))
Base.iterate(r::LanczosResult, ::Val{:vecs}) = (r.vecs, Val(:report))
Base.iterate(r::LanczosResult, ::Val{:report}) = (r.report, Val(:done))
Base.iterate(::LanczosResult, ::Val{:done}) = nothing

"""
    has_converged(result::LanczosResult, λ; atol=1e-6, restol=result.report.eigentol) -> Bool

Return `true` if `result` contains a converged eigenvalue within `atol` of `λ`
whose true residual is below `restol`. Useful for gluing adjacent spectral windows:
ask whether a boundary eigenvalue has already converged in a window.

Reliable when `λ` corresponds to an eigenpair the run actually targeted (e.g. a
spectral extreme, or the target window on POLFED's transformed operator).
"""
function has_converged(result::LanczosResult, λ::Number; atol::Real=1e-6, restol::Real=result.report.eigentol)
    result.converged == 0 && return false
    vals = result.vals
    best_i = 0
    best_d = Inf
    for i in 1:result.converged
        d = abs(vals[i] - λ)
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_d <= atol && result.residuals[best_i] < restol
end
