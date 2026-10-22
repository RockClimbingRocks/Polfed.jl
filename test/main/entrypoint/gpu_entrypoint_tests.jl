# End-to-end GPU tests: the SAME assertions as the CPU entrypoint tests, run against the CUDA
# code path, with the reference always taken from LAPACK on the host.
#
# These are gated, not conditional-by-convention: CUDA.jl is a weak dependency and most machines
# running the suite have no device, so the file must load and report cleanly with no GPU present.
# The gate distinguishes the two "no GPU" cases (CUDA.jl not loadable vs loadable but no working
# device) because they mean different things when a run is unexpectedly skipped.
#
# What the GPU path shares with the CPU path is most of the package: of ~18k lines of source
# exactly one method dispatches on `GPU` (the KPM probe's random vector) and the CUDA extension is
# 39 lines of device/memory queries. Everything else — mapping, DoS, filter, Clenshaw, Lanczos —
# is generic over the array type. So these tests are not re-testing the algorithm; they check that
# the array type flows through it without changing the answer.

using LinearAlgebra, SparseArrays, Random, Test

const _GPU_SKIP = Ref("")

const GPU_OK = try
    @eval using CUDA
    if CUDA.functional()
        true
    else
        _GPU_SKIP[] = "CUDA.jl loaded but no functional device"
        false
    end
catch
    _GPU_SKIP[] = "CUDA.jl not loadable in this environment"
    false
end

# QREM: diag(ε) + hx·Σᵢ σᵢˣ — same fixture as the CPU entrypoint tests.
function _gpu_qrem(L::Int; hx = 0.7)
    D = 2^L
    diagv = randn(D)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, hx)
        end
    end
    return sparse(I, J, V, D, D)
end

_gpu_closest(vals, E, k) = sort(sort(collect(vals); by = λ -> abs(λ - E))[1:k])

if !GPU_OK
    # Register NOTHING when there is no device. Julia's `@test_skip` would report these as
    # "Broken", which conventionally means "a test known to fail" — the wrong claim for hardware
    # that simply is not present. This line is the record that they did not run.
    #
    # Printed rather than logged: runtests.jl raises the ambient logger to Error to hide expected
    # POLFED warnings, and a silently skipped test suite is precisely what must NOT be hidden.
    printstyled(stderr, "GPU end-to-end tests NOT RUN — no CUDA device (", _GPU_SKIP[], ")\n";
                color = :yellow)
else
    @testset "GPU end-to-end vs exact diagonalization — L = $L" for L in (8, 10)
        Random.seed!(20260707)   # same fixture seed as the CPU entrypoint tests: 6/6 at both L
        D = 2^L
        H = _gpu_qrem(L)
        exact = sort(eigvals(Symmetric(Matrix(H))))
        target_E = exact[cld(D, 2)] + 0.13
        howmany = 6

        H_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(H)

        @testset "count mode matches LAPACK" begin
            x0 = CUDA.CuVector(randn(D))
            res = polfed(H_gpu, x0, howmany, target_E)
            vals = sort(collect(Array(res.vals)))
            @test res.converged >= howmany
            for λ in vals
                @test minimum(abs.(exact .- λ)) < 1e-6
            end
            k = min(length(vals), howmany)
            @test maximum(abs.(_gpu_closest(vals, target_E, k) .-
                               _gpu_closest(exact, target_E, k))) < 1e-6
        end

        @testset "eigenvectors satisfy the eigenpair definition on the host" begin
            x0 = CUDA.CuVector(randn(D))
            res = polfed(H_gpu, x0, howmany, target_E)
            vals = Array(res.vals)
            vecs = Array(res.vecs)                       # columns, brought back to the host
            @test size(vecs, 1) == D
            for i in eachindex(vals)
                v = vecs[:, i]
                @test abs(norm(v) - 1) < 1e-8
                @test norm(H * v - vals[i] * v) < 1e-6   # H·v = λ·v, checked against the CPU H
            end
        end

        @testset "window mode returns exactly the enclosed eigenvalues" begin
            i0 = cld(D, 2) - 9; i1 = i0 + 19
            lo = (exact[i0 - 1] + exact[i0]) / 2         # bounds in spectral gaps
            hi = (exact[i1] + exact[i1 + 1]) / 2
            res = polfed(H_gpu, CUDA.CuVector(randn(D)), (:unrescaled, (lo, hi)))
            vals = sort(collect(Array(res.vals)))
            inwin = filter(x -> lo <= x <= hi, vals)
            @test length(inwin) == i1 - i0 + 1
            @test maximum(abs.(inwin .- exact[i0:i1])) < 1e-8
        end

        @testset "GPU and CPU runs agree with each other" begin
            seed = randn(D)
            cpu = polfed(H, copy(seed), howmany, target_E)
            gpu = polfed(H_gpu, CUDA.CuVector(copy(seed)), howmany, target_E)
            vc = sort(collect(cpu.vals)); vg = sort(collect(Array(gpu.vals)))
            n = min(length(vc), length(vg))
            @test n >= howmany
            @test maximum(abs.(vc[1:n] .- vg[1:n])) < 1e-8
            # the reported matvec count is a property of the algorithm, not the device
            @test cpu.report.transform.order == gpu.report.transform.order
        end
    end
end
