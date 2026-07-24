# Shared fixture for the Optimization tests: a hardcoded small QREM Hamiltonian and a
# transpose-free mul! reference. Included first in the Optimization testset so the helpers are
# visible to the per-source test files.

using LinearAlgebra, SparseArrays
import UniPolfed.Optimization: MappingStructure, value_grouping, build_row_mapping,
                            benchmark_mappings, list_mappings, MappingBench

"""
    qrem(L; hx=0.7, T=Float64) -> SparseMatrixCSC

QREM: `diag(ε) + hx·Σᵢ σᵢˣ` on `L` spins (`D = 2^L`). Single off-diagonal value `hx`; every row
has exactly `L` off-diagonal neighbours, the single-bit flips `xor(site, 1<<bit)` — the regime the
optimized kernels target. Real-symmetric (complex-symmetric for complex `T`).
"""
function qrem(L::Int; hx = 0.7, T = Float64)
    D = 2^L
    diagv = randn(T, D)
    I = Int[]; J = Int[]; V = T[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, T(hx))
        end
    end
    return sparse(I, J, V, D, D)
end

"""
    mul_row_ref(H, Xr) -> Yref

Row-layout reference built with `mul!`, one block vector at a time: `Yref[b, :] = H · Xr[b, :]`.
Per-vector column matvecs — no block transpose — so the row kernels are checked against `mul!`
directly.
"""
function mul_row_ref(H, Xr)
    T = eltype(H)
    bs, D = size(Xr)
    Yref = zeros(T, bs, D)
    ycol = zeros(T, D)
    for b in 1:bs
        mul!(ycol, H, collect(@view Xr[b, :]))
        @views Yref[b, :] .= ycol
    end
    return Yref
end
