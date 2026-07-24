using Test
using LinearAlgebra
using Random
using UniPolfed

const LZ3 = UniPolfed.Numerics.Lanczos

function _partial_factorization(A, x0, layout; nsteps=3)
    seed = LZ3._prepare_seed(layout, x0)
    f! = layout isa LZ3.RowLayout ?
         ((Y, X) -> mul!(Y, X, A)) :
         ((Y, X) -> mul!(Y, A, X))
    iterator = LZ3.LanczosIterator(f!, seed, LZ3.FullRO(), layout)
    basis = LZ3.createbasis(32, seed, LZ3.MatrixBasis, LZ3.CPU(), layout)
    factorization = LZ3.lanczositer(iterator, basis, 32, LZ3.CPU())
    for iteration in 2:(nsteps + 1)
        LZ3.lanczositer!(iterator, factorization, iteration) || break
    end
    return factorization
end

function _explicit_ritz_residuals(A, factorization, λ, ϕ)
    basis = Matrix(LZ3.allvecs(factorization.basis))
    if factorization.layout isa LZ3.RowLayout
        vectors = ϕ' * basis
        return [norm(A * collect(view(vectors, i, :)) - λ[i] * collect(view(vectors, i, :)))
                for i in eachindex(λ)]
    end

    vectors = basis * ϕ
    return [norm(A * view(vectors, :, i) - λ[i] * view(vectors, :, i))
            for i in eachindex(λ)]
end

@testset "convergence3" begin
    @testset "Window API and transformed-boundary mode" begin
        legacy = LZ3.Window(0.2, 0.8)
        @test legacy.low == 0.2
        @test legacy.high == 0.8
        @test legacy.edge_value === nothing
        @test legacy.return_brackets

        certified = LZ3.Window(0.2, 0.8, 0.5)
        @test certified.edge_value == 0.5
        @test !certified.return_brackets
        @test_throws ArgumentError LZ3.Window(0.2, 0.8, Inf)

        p = [0.1, 0.4, 0.6, 0.9]
        E = [0.9, 0.7, 0.5, 0.3]
        @test LZ3.pool_indices(certified, p, false) == [4, 3, 2]

        verdict = LZ3.decide(
            LZ3.Highest(certified), p, _ -> true, zeros(4),
            idxs -> (E[idxs], _ -> true),
        )
        @test verdict.done
        @test sort(E[verdict.keep]) == [0.3, 0.5, 0.7]

        negative = LZ3.Window(0.2, 0.8, -0.5)
        negative_p = [-0.9, -0.6, -0.4, -0.1]
        negative_E = [0.3, 0.5, 0.7, 0.9]
        @test LZ3.pool_indices(negative, negative_p, true) == [1, 2, 3]

        negative_verdict = LZ3.decide(
            LZ3.Lowest(negative), negative_p, _ -> true, zeros(4),
            idxs -> (negative_E[idxs], _ -> true),
        )
        @test negative_verdict.done
        @test negative_verdict.keep == [1, 2, 3]

        # With non-leaking transformed bands and genuinely overlapping mapped
        # windows, the selected energy lists have the suffix/prefix seam required
        # by the patch merger. Tail values may appear in any transformed order.
        p_left = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
        E_left = [0.9, 0.8, 0.1, 0.3, 0.5, 0.6]
        left = LZ3.decide(
            LZ3.Highest(LZ3.Window(0.2, 0.7, 0.5)), p_left,
            _ -> true, zeros(6), idxs -> (E_left[idxs], _ -> true),
        )

        p_right = [0.1, 0.2, 0.3, 0.6, 0.8, 0.9]
        E_right = [0.1, 0.3, 0.9, 0.5, 0.6, 0.8]
        right = LZ3.decide(
            LZ3.Highest(LZ3.Window(0.45, 0.85, 0.5)), p_right,
            _ -> true, zeros(6), idxs -> (E_right[idxs], _ -> true),
        )

        left_vals = sort(E_left[left.keep])
        right_vals = sort(E_right[right.keep])
        @test left.done && right.done
        @test left_vals == [0.3, 0.5, 0.6]
        @test right_vals == [0.5, 0.6, 0.8]
        @test left_vals[2:end] == right_vals[1:2]

        # No unconditional debug output from the Window walker.
        mktemp() do _, output
            redirect_stdout(output) do
                LZ3.decide(LZ3.Highest(legacy), p, _ -> true, zeros(4),
                           idxs -> (E[idxs], _ -> true))
            end
            flush(output)
            seekstart(output)
            @test isempty(read(output, String))
        end

        # Infinite bounds waive sides only after at least one mapped pair exists.
        @test !LZ3.is_satisfied(LZ3.Window(-Inf, Inf), Float64[], Float64[], false)
        @test LZ3.is_satisfied(LZ3.Window(-Inf, Inf), Float64[], [0.0], false)
    end

    @testset "current Ritz coupling matches explicit residuals" begin
        Random.seed!(20260721)
        n = 12
        raw = randn(n, n)
        A = Matrix(Symmetric(raw + raw'))

        scalar = _partial_factorization(A, randn(n), LZ3.ColumnLayout(); nsteps=5)
        λs, ϕs, rs = LZ3.calc_ritz_residuals(scalar)
        @test rs ≈ _explicit_ritz_residuals(A, scalar, λs, ϕs) atol=1e-11 rtol=1e-10

        x0 = randn(n, 2)
        columns = _partial_factorization(A, x0, LZ3.ColumnLayout(); nsteps=3)
        λc, ϕc, rc = LZ3.calc_ritz_residuals(columns)
        @test rc ≈ _explicit_ritz_residuals(A, columns, λc, ϕc) atol=1e-11 rtol=1e-10

        rows = _partial_factorization(A, permutedims(x0), LZ3.RowLayout(); nsteps=3)
        λr, ϕr, rr = LZ3.calc_ritz_residuals(rows)
        @test rr ≈ _explicit_ritz_residuals(A, rows, λr, ϕr) atol=1e-11 rtol=1e-10
    end

    @testset "full-dimensional breakdown uses zero current coupling" begin
        Random.seed!(7)
        A = Diagonal(collect(1.0:5.0))
        f!(y, x) = mul!(y, A, x)
        result = LZ3.lanczos(
            f!, randn(5), LZ3.LowestEigs(2);
            maxdim=5, tol=1e-10, eigentol=1e-8,
        )

        @test collect(result.vals) ≈ [1.0, 2.0] atol=1e-10
        @test result.report.complete
        @test result.report.residual == 0.0
    end
end
