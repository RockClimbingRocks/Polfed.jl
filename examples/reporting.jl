# Manual eyeball script — NOT part of the test suite (nothing here asserts anything, and
# `runtests.jl` does not include this directory). It runs a real POLFED workflow and prints the
# report, so report formatting can be reviewed the way a user actually sees it.
#
# Run:
#   julia --project=. examples/reporting.jl
#
# Colors are ON here (stdout is treated as a TTY) because that is the layout being reviewed;
# `show` picks the plain rendering automatically when output is redirected to a file.

using LinearAlgebra, SparseArrays, Random
using Polfed: polfed, MappingConfig

# QREM: diag(ε) + hx·Σᵢσᵢˣ — the standard structured Hermitian test Hamiltonian.
function qrem(L::Int; hx = 0.7)
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

"""Print `title`, then `res.report` with colors on and the convergence table per `details`."""
function show_report(title, res; details::Bool = false)
    println("\n", "="^72)
    println(title)
    println("="^72)
    show(IOContext(stdout, :color => true, :convergence_details => details),
         MIME"text/plain"(), res.report)
    return nothing
end

Random.seed!(20260721)
L = 10
D = 2^L
H = qrem(L)
exact = sort(eigvals(Symmetric(Matrix(H))))

# 1. COUNT target — "give me 20 eigenpairs near this energy".
res_count = polfed(H, randn(D), 20, exact[cld(D, 2)] + 0.13;
                   mc = MappingConfig(optimize_mapping = :shuf))
show_report("COUNT target — polfed(H, x0, howmany, E)", res_count)

# 2. WINDOW target — "give me every eigenpair in this energy interval". Bounds are placed in
#    spectral gaps so the enclosed set is unambiguous.
i0, i1 = cld(D, 2), cld(D, 2) + 19
lo = (exact[i0 - 1] + exact[i0]) / 2
hi = (exact[i1] + exact[i1 + 1]) / 2
res_window = polfed(H, randn(D), (:unrescaled, (lo, hi)))
show_report("WINDOW target — polfed(H, x0, (:unrescaled, (lo, hi)))", res_window;
            details = true)   # also exercise the per-check convergence table

# 3. The result object itself renders a header + the report.
println("\n", "="^72)
println("display(res) — the PolfedResult wrapper")
println("="^72)
display(res_count)
println()
