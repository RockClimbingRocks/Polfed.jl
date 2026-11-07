# End-to-end GPU tests: the SAME assertions as the CPU entrypoint tests, run against the CUDA
# code path, with the reference always taken from LAPACK on the host. Gated on a working CUDA
# device (see `gpu_gate.jl`).
#
# The device path shares the algorithm with the CPU path — mapping, DoS, filter, Clenshaw and
# Lanczos are generic over the array type — so these tests are not re-testing the algorithm. They
# check that every way of running it on the device gives the CPU answer: both block layouts at
# every block size the layout rule distinguishes, every entrypoint, real and complex, and the
# CUSPARSE operator the matrix entrypoints build (which caches its descriptors and workspace).

using LinearAlgebra, SparseArrays, Random, Test
using Polfed: polfed, patched_polfed, full_polfed, double_polfed, MappingConfig,
              ColumnLayout, RowLayout, eigenvector

isdefined(Main, :GPU_OK) || include(joinpath(@__DIR__, "..", "..", "gpu_gate.jl"))

# QREM: diag(ε) + hx·Σᵢ σᵢˣ — the CPU entrypoint fixture — plus an antisymmetric imaginary part for
# a complex T, so Hᵀ ≠ H and a conjugated eigenvector cannot pass.
function _gpu_qrem(L::Int, ::Type{T} = Float64; hx = 0.7) where {T}
    D = 2^L
    diagv = randn(D)
    I = Int[]; J = Int[]; V = T[]
    for i in 1:D
        push!(I, i); push!(J, i); push!(V, diagv[i])
        p = i - 1
        for b in 0:L-1
            push!(I, i); push!(J, xor(p, 1 << b) + 1); push!(V, hx)
        end
    end
    H = sparse(I, J, V, D, D)
    T <: Complex || return H
    P = sprand(D, D, 2 / D)
    return H + im * 0.3 * (P - P')
end

_gpu_closest(vals, E, k) = sort(sort(collect(vals); by = λ -> abs(λ - E))[1:k])

# Every returned pair is an eigenpair of H on the host. `copy` first: a row-layout result is a lazy
# `transpose` of a device view, which `Array` cannot convert in one step.
function _gpu_check_eigenpairs(H, res)
    V = Array(copy(res.vecs))
    for (i, λ) in enumerate(res.vals)
        @test norm(H * V[:, i] - λ * V[:, i]) < 1e-6
    end
end

if !GPU_OK
    gpu_tests_not_run("GPU end-to-end tests")
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
            vals = sort(collect(res.vals))
            @test res.converged >= howmany
            for λ in vals
                @test minimum(abs.(exact .- λ)) < 1e-6
            end
            k = min(length(vals), howmany)
            @test maximum(abs.(_gpu_closest(vals, target_E, k) .-
                               _gpu_closest(exact, target_E, k))) < 1e-6
        end

        @testset "eigenvectors satisfy the eigenpair definition on the host" begin
            res = polfed(H_gpu, CUDA.CuVector(randn(D)), howmany, target_E)
            @test size(res.vecs, 1) == D
            for i in eachindex(res.vals)
                v = Array(eigenvector(res, i))
                @test abs(norm(v) - 1) < 1e-8
                @test norm(H * v - res.vals[i] * v) < 1e-6   # H·v = λ·v, checked against the CPU H
            end
        end

        @testset "window mode returns exactly the enclosed eigenvalues" begin
            i0 = cld(D, 2) - 9; i1 = i0 + 19
            lo = (exact[i0 - 1] + exact[i0]) / 2         # bounds in spectral gaps
            hi = (exact[i1] + exact[i1 + 1]) / 2
            res = polfed(H_gpu, CUDA.CuVector(randn(D)), (:unrescaled, (lo, hi)))
            vals = sort(collect(res.vals))
            inwin = filter(x -> lo <= x <= hi, vals)
            @test length(inwin) == i1 - i0 + 1
            @test maximum(abs.(inwin .- exact[i0:i1])) < 1e-8
        end

        @testset "GPU and CPU runs agree with each other" begin
            seed = randn(D)
            cpu = polfed(H, copy(seed), howmany, target_E)
            gpu = polfed(H_gpu, CUDA.CuVector(copy(seed)), howmany, target_E)
            vc = sort(collect(cpu.vals)); vg = sort(collect(gpu.vals))
            n = min(length(vc), length(vg))
            @test n >= howmany
            @test maximum(abs.(vc[1:n] .- vg[1:n])) < 1e-8
            # The filter order comes from the stochastic DoS, and device probes come from the
            # device generator (not the host `rng`), so the two orders agree only up to its error.
            @test isapprox(cpu.report.transform.order, gpu.report.transform.order; rtol = 0.05)
        end
    end

    @testset "GPU matrix entrypoint: every layout and block size, $T" for T in (Float64, ComplexF64)
        Random.seed!(20260707)
        L = 8; D = 2^L
        H = _gpu_qrem(L, T)
        exact = sort(eigvals(Hermitian(Matrix(H))))
        target_E = exact[cld(D, 2)] + 0.13
        howmany = 6
        H_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(H)
        # bs = 1: scalar Lanczos (auto → column) and a 1-row block; bs = 4: both layouts below the
        # auto threshold; bs = 16: auto → row, SpMM ALG3; bs = 32: row, ALG2 for Float64 and ALG3
        # for ComplexF64 (`_row_alg2_from`). (Wider blocks need
        # a larger D: the Krylov budget is capped at the dimension, 256 here.)
        @testset "$(something(layout, "auto")) layout, bs = $bs" for (layout, bs) in (
                (nothing, 1), (RowLayout(), 1), (ColumnLayout(), 4), (RowLayout(), 4),
                (nothing, 16), (ColumnLayout(), 16), (RowLayout(), 32))
            x0 = bs == 1 ? CUDA.CuVector(randn(T, D)) : CUDA.CuArray(randn(T, D, bs))
            res = polfed(H_gpu, x0, howmany, target_E; mc = MappingConfig(layout = layout))
            @test res.vals isa Vector{Float64}
            @test maximum(abs.(_gpu_closest(res.vals, target_E, howmany) .-
                               _gpu_closest(exact, target_E, howmany))) < 1e-6
            _gpu_check_eigenpairs(H, res)
        end

        @testset "operator entrypoint, bs = $bs" for bs in (1, 4)
            f! = (Y, X) -> mul!(Y, H_gpu, X)
            x0 = bs == 1 ? CUDA.CuVector(randn(T, D)) : CUDA.CuArray(randn(T, D, bs))
            res = polfed(f!, x0, howmany, target_E)
            @test maximum(abs.(_gpu_closest(res.vals, target_E, howmany) .-
                               _gpu_closest(exact, target_E, howmany))) < 1e-6
            _gpu_check_eigenpairs(H, res)
        end

        @testset "patched, full and double POLFED" begin
            res = patched_polfed(H_gpu, CUDA.CuVector(randn(T, D)), 40, 12, target_E)
            @test maximum(abs.(_gpu_closest(res.vals, target_E, 40) .-
                               _gpu_closest(exact, target_E, 40))) < 1e-6
            full = full_polfed(H_gpu, CUDA.CuVector(randn(T, D)), 64)
            @test maximum(abs.(sort(collect(full.vals)) .- exact)) < 1e-6
            E1, E2 = exact[cld(D, 3)] + 0.07, exact[2cld(D, 3)] + 0.05
            r1, r2 = double_polfed(H_gpu, CUDA.CuVector(randn(T, D)), 5, E1, E2)
            @test maximum(abs.(_gpu_closest(r1.vals, E1, 5) .- _gpu_closest(exact, E1, 5))) < 1e-6
            @test maximum(abs.(_gpu_closest(r2.vals, E2, 5) .- _gpu_closest(exact, E2, 5))) < 1e-6
        end
    end

    @testset "GPU layout rule and device checks" begin
        @test Polfed._matrix_layout(nothing, Polfed.GPU(), 1) == ColumnLayout()
        @test Polfed._matrix_layout(nothing, Polfed.GPU(), Polfed.Defaults.polfed.gpu_row_layout_from) ==
              RowLayout()
        @test Polfed._matrix_layout(ColumnLayout(), Polfed.GPU(), 64) == ColumnLayout()
        H = _gpu_qrem(6)
        @test_throws ArgumentError polfed(H, CUDA.CuVector(randn(size(H, 1))), 2, 0.1)
        @test_throws ArgumentError polfed(CUDA.CUSPARSE.CuSparseMatrixCSR(H), randn(size(H, 1)), 2, 0.1)
        @test_throws ArgumentError polfed(CUDA.CUSPARSE.CuSparseMatrixCSR(H), CUDA.CuVector(randn(size(H, 1))),
                                          2, 0.1; mc = MappingConfig(optimize_mapping = :auto))
    end

    # The cached CUSPARSE operator against host arithmetic, with the input and output rotating
    # through different buffers and α/β changing between calls — as the Clenshaw recurrence does.
    @testset "CUSPARSE operator, $(nameof(typeof(layout))), shape $sz" for (layout, sz) in (
            (RowLayout(), (1, 512)), (RowLayout(), (8, 512)), (RowLayout(), (40, 512)),
            (ColumnLayout(), (512, 1)), (ColumnLayout(), (512, 8)))
        Random.seed!(1)
        H = sprandn(ComplexF64, 512, 512, 12 / 512); H = H + H'
        op = Polfed._gpu_matrix_operator(CUDA.CUSPARSE.CuSparseMatrixCSR(H), layout)
        apply(X) = layout isa RowLayout ? permutedims(H * permutedims(X)) : H * X
        bufs = [CUDA.CuArray(randn(ComplexF64, sz...)) for _ in 1:3]
        for rep in 1:6
            X, Y = bufs[mod1(rep, 3)], bufs[mod1(rep + 1, 3)]
            Xh, Yh = Array(X), Array(Y)
            α, β = isodd(rep) ? (2.0, -1.0) : (1.0, 0.0)
            Polfed._axpby!(op, Y, X, α, β)
            @test Array(Y) ≈ α .* apply(Xh) .+ β .* Yh
        end
        # a strided 1 × D row of a larger block — how a row-layout eigenvector reaches the map-back
        if layout isa RowLayout
            B = CUDA.CuArray(randn(ComplexF64, 4, 512)); y = CUDA.zeros(ComplexF64, 1, 512)
            op(y, view(B, 3:3, :))
            @test Array(y) ≈ permutedims(H * Array(B)[3, :])
        end
    end
end
