using Test
using UniPolfed

function gaussian_dos(peak, width, dim; N = 80, M = 8000)
    base(x) = exp(-((x - peak)^2) / width^2)

    # Normalize the plain Gaussian so ∫_{-1}^{1} ρ(x) dx = dim.
    norm = 0.0
    for m in 1:M
        θ = (m - 0.5) * π / M
        norm += base(cos(θ)) * sin(θ)
    end
    norm *= π / M

    ρ = x -> dim * base(x) / norm

    # resolve_interval integrates Chebyshev coefficients c_k for
    # ρ(x) = (Σ c_k T_k(x)) / (π√(1-x²)). Project π√(1-x²)ρ(x) onto T_k.
    coefficients = zeros(Float64, N)
    for m in 1:M
        θ = (m - 0.5) * π / M
        x = cos(θ)
        weighted_density = π * sin(θ) * ρ(x)
        coefficients[1] += weighted_density
        for k in 1:N-1
            coefficients[k + 1] += weighted_density * cos(k * θ)
        end
    end
    coefficients[1] /= M
    coefficients[2:end] .*= 2 / M

    moments = copy(coefficients)
    return UniPolfed.DoSResult(moments, coefficients, ρ)
end

state_count(dos, interval) =
    UniPolfed.integrate_chebyshev_weighted(dos.coefficients, interval.left, interval.right) / π


@testset "resolve_interval" begin
    
    @testset "centered target grows a symmetric window" begin
        dim = 10
        howmany = 2
        peak = 0.2
        dos = gaussian_dos(peak, 0.25, dim)
        interval = UniPolfed.resolve_interval(UniPolfed.Target(peak, howmany, :center), dos)

        @test (interval.left + interval.right) / 2 ≈ peak atol = 1e-8
        @test state_count(dos, interval) ≈ howmany rtol = 1e-4
    end

    @testset "left target keeps the target as the left endpoint" begin
        dim = 10
        howmany = 2
        peak = -0.2
        dos = gaussian_dos(peak, 0.25, dim)
        interval = UniPolfed.resolve_interval(UniPolfed.Target(peak, howmany, :left), dos)

        @test interval.left ≈ peak atol = 1e-8
        @test state_count(dos, interval) ≈ howmany rtol = 1e-4
    end

    @testset "right target keeps the target as the right endpoint" begin
        dim = 10
        howmany = 2
        peak = 0.2
        dos = gaussian_dos(peak, 0.25, dim)
        interval = UniPolfed.resolve_interval(UniPolfed.Target(peak, howmany, :right), dos)

        @test interval.right ≈ peak atol = 1e-8
        @test state_count(dos, interval) ≈ howmany rtol = 1e-4
    end

    @testset "unavailable state count returns widest clamped window" begin
        dos = gaussian_dos(0.0, 0.25, 10)

        @test_logs (:warn, r"widest window") begin
            interval = UniPolfed.resolve_interval(UniPolfed.Target(0.8, 100, :center), dos)
            @test interval.left ≈ -0.2
            @test interval.right ≈ 1.0
        end
    end

    @testset "invalid target placement is rejected" begin
        @test_throws ArgumentError UniPolfed.Target(0.0, 1, :middle)
        @test_throws ArgumentError UniPolfed._window_reshape(0.0, :middle, 0.1)
    end
end
