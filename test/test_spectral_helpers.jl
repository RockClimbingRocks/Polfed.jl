using LinearAlgebra

const CORE = Polfed.PolfedCore

identity_map!(Y, X) = copyto!(Y, X)

function identity_mapping_plan(dim::Integer=4)
    x0 = ones(Float64, dim)
    mapping = Polfed.MappingConfig(
        parallel_strategy=Polfed.NoParallel(),
        Emin=-1.0,
        Emax=1.0,
    )

    return CORE.build_mapping_plan(mapping, identity_map!, x0, Polfed.CPU())
end

@testset "DoS kernels" begin
    @test CORE.Dirichlet(0, 8) == 1
    @test CORE.Dirichlet(5, 8) == 1

    @test CORE.Fejer(0, 10) == 1
    @test CORE.Fejer(4, 10) ≈ 0.6
    @test CORE.Fejer(10, 10) == 0

    @test CORE.Lorentz(0, 10) ≈ 1
    @test CORE.Lorentz(10, 10) ≈ 0
    @test CORE.LanczosK(0, 10) == 1
    @test CORE.WangZunger(0, 10) == 1
    @test CORE.WangZunger(5, 10) < CORE.WangZunger(0, 10)
    @test CORE.Jackson(0, 10) ≈ 1

    @test CORE.get_kernel(:Fejer) === CORE.Fejer
    @test CORE.get_kernel(:Dirichlet) === CORE.Dirichlet
    @test_throws ArgumentError CORE.get_kernel(:UnknownKernel)
end

@testset "DoS moments" begin
    α = [0.6, 0.8]
    μs = zeros(5)
    vecs = [similar(α) for _ in 1:3]

    CORE.trace!(identity_map!, α, μs, vecs)

    @test μs ≈ ones(5)

    moments = CORE.dos_moments(identity_map!, 5, 3, 4, Float64, Polfed.CPU())

    @test moments ≈ fill(4.0, 5)

    mapping_plan = identity_mapping_plan(4)
    fact = CORE.FactorizationConfigFull(Polfed.FactorizationConfig(), ones(4), 1)
    dos = CORE.DoSConfigFull(Polfed.DoSConfig(N=5, R=3, kernel=:Dirichlet))

    CORE.getdos!(dos, mapping_plan, fact, Polfed.CPU())

    @test dos.ρ isa Function
    @test isfinite(dos.ρ(0.0))
    @test dos.ρ(0.0) ≈ 4 / π
end

@testset "spectral transform helpers" begin
    @test CORE.bisection(x -> x - 0.25, 0.0, 1.0; tol=1e-10) ≈ 0.25 atol=1e-8
    @test_throws ErrorException CORE.bisection(x -> x^2 + 1, -1.0, 1.0)

    @test CORE.analytical_orderofexpansion(0.0, 0.5) ≈ 2.65499 / 0.5
    @test CORE.analytical_orderofexpansion(0.0, 0.25) > CORE.analytical_orderofexpansion(0.0, 0.5)
    @test CORE.analytical_orderofexpansion(0.8, 0.5) < CORE.analytical_orderofexpansion(0.0, 0.5)

    mapping_plan = identity_mapping_plan(4)
    ρ(x) = 1.0

    bounds_from_dos = CORE.build_transform_plan(Polfed.TransformConfig(), mapping_plan, 1, :middle)
    CORE.getbounds_from_dos!(bounds_from_dos, ρ, :mean)

    @test bounds_from_dos.left ≈ -0.5 atol=1e-7
    @test bounds_from_dos.right ≈ 0.5 atol=1e-7

    bounds_from_order = CORE.build_transform_plan(
        Polfed.TransformConfig(order=16, cutoff=0.2),
        mapping_plan,
        1,
        :middle,
    )
    CORE.getbounds_from_K!(bounds_from_order, :mean)

    @test -1 <= bounds_from_order.left < bounds_from_order.target
    @test bounds_from_order.target < bounds_from_order.right <= 1
    @test bounds_from_order.left ≈ -bounds_from_order.right atol=1e-7

    explicit_order = CORE.build_transform_plan(
        Polfed.TransformConfig(order=12),
        mapping_plan,
        1,
        :middle,
    )
    CORE.getorderofexpansion!(explicit_order)

    @test explicit_order.order == 12

    resolved_order = CORE.build_transform_plan(
        Polfed.TransformConfig(left=-0.2, right=0.2, cutoff=0.2, order_safety_factor=1.0),
        mapping_plan,
        1,
        :middle,
    )
    CORE.getorderofexpansion!(resolved_order)

    @test resolved_order.order isa Integer
    @test resolved_order.order >= 10

    overconstrained = CORE.build_transform_plan(
        Polfed.TransformConfig(left=-0.2, right=0.2, order=12),
        mapping_plan,
        1,
        :middle,
    )

    @test_throws ErrorException CORE.getbounds!(overconstrained, mapping_plan, ρ, :mean)
end
