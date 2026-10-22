# Shared fixture for the report tests: one real POLFED run (matrix entrypoint, row-native,
# :universal kernel) whose PolfedResult + Report the behavioral tests below inspect. Running polfed
# once here keeps the report testset fast.

using LinearAlgebra, SparseArrays, Random
using UniPolfed: polfed, MappingConfig, PolfedResult

# QREM: diag(ε) + hx·Σᵢσᵢˣ — the structured Hermitian matrix used across the report tests.
function _report_qrem(L::Int; hx = 0.7)
    D = 2^L
    dv = randn(D)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, dv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, hx)
        end
    end
    return sparse(I, J, V, D, D)
end

Random.seed!(20260707)
const REPORT_L       = 8
const REPORT_D       = 2^REPORT_L
const REPORT_H       = _report_qrem(REPORT_L)
const REPORT_EXACT   = sort(eigvals(Symmetric(Matrix(REPORT_H))))
const REPORT_HOWMANY = 6
const REPORT_TARGET  = REPORT_EXACT[cld(REPORT_D, 2)] + 0.13
const RES            = polfed(REPORT_H, randn(REPORT_D), REPORT_HOWMANY, REPORT_TARGET;
                              mc = MappingConfig(optimize_mapping = :universal))

# Render a Report/PolfedResult to a String with explicit IO context (color / details).
_render(r; color::Bool = false, details::Bool = false) =
    sprint(io -> show(IOContext(io, :color => color, :convergence_details => details),
                      MIME"text/plain"(), r))
