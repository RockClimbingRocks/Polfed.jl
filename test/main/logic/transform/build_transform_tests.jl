using LinearAlgebra
using Test
using Polfed

function with_silent_polfed_logs(f)
    old_verbosity = Polfed.Common.verbosity[]
    Polfed.Common.verbosity[] = Polfed.Common.POLFED_SILENT_LEVEL
    try
        return f()
    finally
        Polfed.Common.verbosity[] = old_verbosity
    end
end

@testset "build_transform" begin
    @testset "returns an in-place transformed operator" begin
        D = 6
        H = Diagonal(range(-1.0, 1.0; length = D))
        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(Float64, D)

        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(
                Emin = -1.0,
                Emax = 1.0,
                parallel_strategy = Polfed.BlockWise(),
            ),
            f!,
            x0,
        )
        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.2, 0.2);
            order = 8,
        )

        f!_transformed = Polfed.build_transform(filter, mp)
        X = collect(1.0:D)
        Y = similar(X)

        @test f!_transformed(Y, X) === nothing
        @test size(Y) == size(X)
        @test eltype(Y) === eltype(X)
        @test all(isfinite, Y)
    end

    @testset "matches scalar filter on diagonal operator" begin
        λ = [-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)

        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(Float64, length(λ))

        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(Emin = -1.0, Emax = 1.0),
            f!,
            x0,
        )

        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        f!_transformed = Polfed.build_transform(filter, mp)

        X = collect(1.0:length(λ))
        Y = similar(X)

        f!_transformed(Y, X)

        P = Polfed._scalar_filter(filter)
        expected = P.(λ) .* X

        @test Y ≈ expected
    end

    @testset "uses the combined polynomial order for DoubleDeltaFilter" begin
        λ = [-0.9, -0.75, -0.4, 0.0, 0.25, 0.8]
        H = Diagonal(λ)
        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(Float64, length(λ))

        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(
                Emin=-1.0,
                Emax=1.0,
                parallel_strategy=Polfed.BlockWise(),
            ),
            f!,
            x0,
        )
        filter = Polfed.build_filter(
            Polfed.DoubleDeltaFilter,
            Polfed.Interval(-0.80, -0.70),
            Polfed.Interval(0.20, 0.30);
            order1=8,
            order2=13,
        )

        @test Polfed.polynomial_order(filter) === 13

        transform! = Polfed.build_transform(filter, mp)
        X = collect(1.0:length(λ))
        Y = similar(X)
        transform!(Y, X)

        expected = Polfed._scalar_filter(filter).(λ) .* X
        @test Y ≈ expected
    end

    @testset "rejects an unresolved filter" begin
        H = Diagonal([-1.0, 0.0, 1.0])
        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(3)
        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(Emin=-1.0, Emax=1.0),
            f!,
            x0,
        )
        unresolved = Polfed.DeltaUpFilter(Polfed.Interval(-0.2, 0.2))

        @test Polfed.polynomial_order(unresolved) < 0
        @test_throws ArgumentError Polfed.build_transform(unresolved, mp)
    end

    @testset "matches scalar filter on diagonal operator for block input" begin
        λ = [-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)

        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(Float64, length(λ), 2)

        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(
                Emin = -1.0,
                Emax = 1.0,
                parallel_strategy = Polfed.BlockWise(),
            ),
            f!,
            x0,
        )
        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        f!_transformed = Polfed.build_transform(filter, mp)
        X = reshape(collect(1.0:(2 * length(λ))), length(λ), 2)
        Y = similar(X)

        f!_transformed(Y, X)

        P = Polfed._scalar_filter(filter)
        expected = P.(λ) .* X

        @test Y ≈ expected
    end

    @testset "matches scalar filter on diagonal operator for complex input" begin
        λ = [-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)

        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(ComplexF64, length(λ))

        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(Emin = -1.0, Emax = 1.0),
            f!,
            x0,
        )
        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        f!_transformed = Polfed.build_transform(filter, mp)
        X = ComplexF64[1 + 2im, -2 + 0.5im, 3 - im, -4 - 2im, 5 + 0im]
        Y = similar(X)

        f!_transformed(Y, X)

        P = Polfed._scalar_filter(filter)
        expected = P.(λ) .* X

        @test eltype(Y) === ComplexF64
        @test Y ≈ expected
    end

    @testset "preserves Float32 input type" begin
        λ = Float32[-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)

        f!(Y, X) = (mul!(Y, H, X); nothing)
        x0 = ones(Float32, length(λ))

        mp = Polfed.build_mapping_plan(
            Polfed.MappingConfig(Emin = -1.0f0, Emax = 1.0f0),
            f!,
            x0,
        )
        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        f!_transformed = Polfed.build_transform(filter, mp)
        X = Float32[1, -2, 3, -4, 5]
        Y = similar(X)

        f!_transformed(Y, X)

        P = Polfed._scalar_filter(filter)
        expected = Float32.(P.(λ)) .* X

        @test eltype(Y) === Float32
        @test Y ≈ expected rtol = 1f-5
    end

    @testset "right and left multiplication agree up to transpose" begin
        λ = [-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)
        D = length(λ)
        bs = 2

        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        f_col!(Y, X) = (mul!(Y, H, X); nothing)
        x0_col = ones(Float64, D, bs)
        mp_col = Polfed.build_mapping_plan(
            Polfed.MappingConfig(Emin = -1.0, Emax = 1.0, parallel_strategy = Polfed.BlockWise()),
            f_col!,
            x0_col,
        )
        transform_col! = Polfed.build_transform(filter, mp_col)

        f_row!(Y, X) = (mul!(Y, X, H); nothing)
        x0_row = permutedims(x0_col)
        mp_row = Polfed.build_mapping_plan(
            Polfed.MappingConfig(
                Emin = -1.0,
                Emax = 1.0,
                f!_rescaled = f_row!,
                parallel_strategy = Polfed.BlockWise(),
            ),
            f_row!,
            x0_row,
        )
        transform_row! = Polfed.build_transform(filter, mp_row)

        X_col = reshape(collect(1.0:(D * bs)), D, bs)
        X_row = permutedims(X_col)
        Y_col = similar(X_col)
        Y_row = similar(X_row)

        transform_col!(Y_col, X_col)
        transform_row!(Y_row, X_row)

        @test permutedims(Y_row) ≈ Y_col
    end

    @testset "parallel strategies agree for vector input" begin
        λ = [-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)
        D = length(λ)
        f!(Y, X) = (mul!(Y, H, X); nothing)
        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        function transformed_with(strategy)
            x0 = ones(Float64, D)
            mp = Polfed.build_mapping_plan(
                Polfed.MappingConfig(
                    Emin = -1.0,
                    Emax = 1.0,
                    parallel_strategy = strategy,
                ),
                f!,
                x0,
            )
            transform! = Polfed.build_transform(filter, mp)
            X = collect(1.0:D)
            Y = similar(X)
            transform!(Y, X)
            return Y
        end

        Y_blockwise = transformed_with(Polfed.BlockWise())
        Y_vectorwise = transformed_with(Polfed.VectorWise())

        @test Y_vectorwise ≈ Y_blockwise
    end

    @testset "parallel strategies agree for block input" begin
        λ = [-0.8, -0.3, 0.0, 0.4, 0.9]
        H = Diagonal(λ)
        D = length(λ)
        bs = 2
        f!(Y, X) = (mul!(Y, H, X); nothing)
        filter = Polfed.build_filter(
            Polfed.DeltaUpFilter,
            Polfed.Interval(-0.1, 0.1);
            order = 12,
        )

        function transformed_with(strategy)
            x0 = ones(Float64, D, bs)
            mp = with_silent_polfed_logs() do
                Polfed.build_mapping_plan(
                    Polfed.MappingConfig(
                        Emin = -1.0,
                        Emax = 1.0,
                        parallel_strategy = strategy,
                    ),
                    f!,
                    x0,
                )
            end
            transform! = Polfed.build_transform(filter, mp)
            X = reshape(collect(1.0:(D * bs)), D, bs)
            Y = similar(X)
            transform!(Y, X)
            return Y
        end

        Y_blockwise = transformed_with(Polfed.BlockWise())
        Y_vectorwise = transformed_with(Polfed.VectorWise())

        @test Y_vectorwise ≈ Y_blockwise
    end
    
end
