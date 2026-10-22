using Test
using Polfed

@testset "build_expansion" begin
    @testset "normalize_clenshaw_recurrence accepts 5-argument recurrence" begin
        called = Ref(false)
        recurrence5!(b1, b2, _b3, c, X) = begin
            called[] = true
            @. b1 = b2 + c * X
            return nothing
        end

        x0 = ones(Float64, 4)
        recurrence! = Polfed.normalize_clenshaw_recurrence(
            recurrence5!,
            x0,
            Polfed.BlockWise(),
        )

        b1 = zeros(4)
        b2 = fill(2.0, 4)
        b3 = fill(3.0, 4)
        X = fill(5.0, 4)

        @test recurrence!(b1, b2, b3, 0.5, 7, X) === nothing
        @test called[]
        @test b1 ≈ b2 .+ 0.5 .* X
    end

    @testset "normalize_clenshaw_recurrence accepts 6-argument recurrence" begin
        seen_k = Ref(0)
        recurrence6!(b1, b2, _b3, c, k, X) = begin
            seen_k[] = k
            @. b1 = b2 + k * c * X
            return nothing
        end

        x0 = ones(Float64, 4)
        recurrence! = Polfed.normalize_clenshaw_recurrence(
            recurrence6!,
            x0,
            Polfed.BlockWise(),
        )

        b1 = zeros(4)
        b2 = fill(2.0, 4)
        b3 = fill(3.0, 4)
        X = fill(5.0, 4)

        @test recurrence!(b1, b2, b3, 0.5, 7, X) === nothing
        @test seen_k[] == 7
        @test b1 ≈ b2 .+ 7 .* 0.5 .* X
    end

    @testset "normalize_clenshaw_recurrence rejects invalid recurrence" begin
        x0 = ones(Float64, 4)
        bad_recurrence!(_b1, _b2) = nothing

        @test_throws ArgumentError Polfed.normalize_clenshaw_recurrence(
            bad_recurrence!,
            x0,
            Polfed.BlockWise(),
        )
    end
    @testset "_operator_expansion picks the evaluator from the operator" begin
        C = Polfed.Numerics.Clenshaw
        H = Polfed.Models.xxz_hamiltonian(8, 4, 1.0, 0.55, 0.0; boundary = :periodic,
                                             field = 0.0, use_sparse = true)
        D, bs, a, b = size(H, 1), 2, 4.0, 0.25
        coef = k -> 1 / (1 + k)
        X = rand(bs, D)
        ws() = [rand(bs, D) for _ in 1:3]
        ref(op, K) = C.ClenshawExpansion(:Chebyshev, coef, K, op)(zeros(bs, D), X, ws())

        gen = Polfed._generic_row_rescaled(H, a, b)
        e = Polfed._operator_expansion(:Chebyshev, coef, 30, gen)
        @test e isa C.AxpbyClenshaw
        @test e(zeros(bs, D), X, ws()) ≈ ref(gen, 30) rtol = 1e-12

        opt = Polfed.build_optimized_mapping(H, :universal_row_read, Val(bs), a, b).f!_rescaled
        @test opt isa Polfed.RangeKernelOperator
        e = Polfed._operator_expansion(:Chebyshev, coef, 30, opt)
        @test e isa C.RangeClenshaw
        @test e(zeros(bs, D), X, ws()) == ref(opt.whole, 30)   # same arithmetic, bit for bit

        plain = (Y, Z) -> (Y .= 0.5 .* Z; nothing)
        @test Polfed._operator_expansion(:Chebyshev, coef, 30, plain) isa C.ClenshawExpansion
    end
end
