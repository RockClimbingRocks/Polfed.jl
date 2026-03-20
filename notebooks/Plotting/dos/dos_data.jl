using LinearAlgebra
using Random
using SparseArrays
using HDF5: h5open, create_group

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
if !isdefined(Main, :Polfed)
    include(joinpath(PROJECT_ROOT, "src", "Polfed.jl"))
end
using .Polfed
const PolfedCore = Polfed.PolfedCore


function unique_periodic_bonds(L::Int, distance::Int)
    bonds = Set{Tuple{Int, Int}}()
    for i in 1:L
        j = mod1(i + distance, L)
        a, b = minmax(i, j)
        push!(bonds, (a, b))
    end
    return sort!(collect(bonds))
end


@inline function add_exchange_bond!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    bmap::Dict{Int, Int},
    state::Int,
    col::Int,
    i::Int,
    j::Int,
    J::Float64,
)
    si = (state >> (i - 1)) & 1
    sj = (state >> (j - 1)) & 1
    sz_i = si == 1 ? -0.5 : 0.5
    sz_j = sj == 1 ? -0.5 : 0.5
    diag_contrib = J * sz_i * sz_j

    if si != sj
        flipped = xor(xor(state, 1 << (i - 1)), 1 << (j - 1))
        push!(rows, bmap[flipped])
        push!(cols, col)
        push!(vals, 0.5 * J)
    end

    return diag_contrib
end


function construct_disordered_j1j2_matrix(
    L::Int,
    N_particles::Int,
    disorder::AbstractVector{<:Real};
    J1::Float64 = 1.0,
    J2::Float64 = 0.55,
)
    0 <= N_particles <= L || throw(ArgumentError("N_particles must satisfy 0 <= N_particles <= L."))
    length(disorder) == L || throw(ArgumentError("Disorder vector must have length L."))

    basis = [b for b in 0:(1 << L) - 1 if count_ones(b) == N_particles]
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))

    bonds_J1 = unique_periodic_bonds(L, 1)
    bonds_J2 = unique_periodic_bonds(L, 2)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (col, state) in enumerate(basis)
        diag = 0.0

        for site in 1:L
            occ = (state >> (site - 1)) & 1
            sz = occ == 1 ? -0.5 : 0.5
            diag += disorder[site] * sz
        end

        for (i, j) in bonds_J1
            diag += add_exchange_bond!(rows, cols, vals, bmap, state, col, i, j, J1)
        end

        for (i, j) in bonds_J2
            diag += add_exchange_bond!(rows, cols, vals, bmap, state, col, i, j, J2)
        end

        push!(rows, col)
        push!(cols, col)
        push!(vals, diag)
    end

    return sparse(rows, cols, vals, dim, dim)
end


function compute_kpm_dos(
    mat::SparseMatrixCSC{Float64, Int},
    eigvals_exact::Vector{Float64};
    moments::Int = 512,
    R::Int = 250,
    grid_size::Int = 2000,
    rng::AbstractRNG = Random.default_rng(),
)
    dim = size(mat, 1)
    x0 = randn(rng, Float64, dim)
    x0 ./= norm(x0)

    f! = (Y, X) -> mul!(Y, mat, X)
    mapping_cfg = Polfed.MappingConfig(
        Emin = first(eigvals_exact),
        Emax = last(eigvals_exact),
    )

    map_plan = PolfedCore.build_mapping_plan(mapping_cfg, f!, x0, Polfed.CPU())
    fact_full = PolfedCore.FactorizationConfigFull(Polfed.FactorizationConfig(), x0, 1)
    dos_full = PolfedCore.DoSConfigFull(Polfed.DoSConfig(N = moments, R = R, kernel = :Jackson))
    PolfedCore.getdos!(dos_full, map_plan, fact_full, Polfed.CPU())

    x_grid = collect(range(-0.999, 0.999; length = grid_size))
    energy_grid = map_plan.a .* x_grid .+ map_plan.b
    rho_fn = getfield(dos_full, Symbol("\u03C1"))
    dos_rescaled = rho_fn.(x_grid)
    dos_energy = dos_rescaled ./ abs(map_plan.a)

    return (
        x_grid = x_grid,
        energy_grid = energy_grid,
        dos_rescaled = dos_rescaled,
        dos_energy = dos_energy,
        Emin = map_plan.Emin,
        Emax = map_plan.Emax,
        scale_a = map_plan.a,
        shift_b = map_plan.b,
    )
end


function run_disordered_j1j2_scan(;
    L::Int = 20,
    N::Int = 5,
    J1::Float64 = 1.0,
    J2::Float64 = 0.55,
    R::Int = 1500,
    N_moments::Vector{Int} = [25, 50,100,200, 400, 800, 1600, 32_000],
    dos_grid_size::Int = 2000,
    disorder_seed::Int = 1234,
    output_file::String = joinpath(@__DIR__, "disordered_j1j2_dos_L$(L)_N$(N).h5"),
)
    1 <= N <= (L - 1) || throw(ArgumentError("N must satisfy 1 <= N <= L-1."))
    moments_list = unique(sort(N_moments))
    isempty(moments_list) && throw(ArgumentError("N_moments cannot be empty."))
    all(m -> m >= 2, moments_list) || throw(ArgumentError("All N_moments values must be >= 2."))

    julia_threads = Threads.nthreads()
    julia_threads == 1 && @warn "Running with one Julia thread. Set JULIA_NUM_THREADS>1 for parallel KPM."
    BLAS.set_num_threads(julia_threads)
    blas_threads = BLAS.get_num_threads()

    disorder = 2.0 .* rand(L) .- 1.0

    println("Running disordered J1-J2 scan")
    println("L = $L, N = $N, J1 = $J1, J2 = $J2, R = $R")
    println("N_moments = $moments_list")
    println("Julia threads = $julia_threads, BLAS threads = $blas_threads")
    println("Disorder sampled uniformly on [-1, 1] with seed = $disorder_seed")

    mat_sparse = construct_disordered_j1j2_matrix(
        L,
        N,
        disorder;
        J1 = J1,
        J2 = J2,
    )

    mat = Matrix(mat_sparse)
    mat = 0.5 .* (mat .+ transpose(mat))
    eigvals_exact = real(eigvals(mat))
    sort!(eigvals_exact)

    h5open(output_file, "w") do h5
        h5["L"] = L
        h5["N"] = N
        h5["J1"] = J1
        h5["J2"] = J2
        h5["R"] = R
        h5["N_moments"] = moments_list
        h5["dos_grid_size"] = dos_grid_size
        h5["julia_threads"] = julia_threads
        h5["blas_threads"] = blas_threads
        h5["disorder"] = disorder
        h5["disorder_seed"] = disorder_seed

        group_N = create_group(h5, "N_$(N)")
        group_N["N_particles"] = N
        group_N["hilbert_dim"] = size(mat_sparse, 1)
        group_N["nnz"] = nnz(mat_sparse)
        group_N["eigvals_exact"] = eigvals_exact

        for m in moments_list
            println("  -> KPM moments = $m")
            kpm_rng = MersenneTwister(disorder_seed + 10_000 + N + m)
            kpm = compute_kpm_dos(
                mat_sparse,
                eigvals_exact;
                moments = m,
                R = R,
                grid_size = dos_grid_size,
                rng = kpm_rng,
            )

            group_m = create_group(group_N, "moments_$(m)")
            group_m["N_moments"] = m
            group_m["energy_grid"] = kpm.energy_grid
            group_m["dos_kpm"] = kpm.dos_energy
            group_m["x_grid_rescaled"] = kpm.x_grid
            group_m["dos_kpm_rescaled"] = kpm.dos_rescaled
            group_m["Emin"] = kpm.Emin
            group_m["Emax"] = kpm.Emax
            group_m["scale_a"] = kpm.scale_a
            group_m["shift_b"] = kpm.shift_b
        end
    end

    println("Saved results to: $output_file")
    return output_file
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_disordered_j1j2_scan()
end
