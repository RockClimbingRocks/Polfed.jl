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


function construct_xxz_spin_sector(L::Int, delta::Real, Nup::Int)
    2 <= L || throw(ArgumentError("L must be >= 2."))
    0 <= Nup <= L || throw(ArgumentError("Nup must satisfy 0 <= Nup <= L."))

    basis = [b for b in 0:(2^L - 1) if count_ones(b) == Nup]
    dim = length(basis)
    bmap = Dict(b => i for (i, b) in enumerate(basis))
    rows, cols, vals = Int[], Int[], Float64[]

    for (col, state) in enumerate(basis)
        for i in 1:L
            j = mod1(i + 1, L)
            si = (state >> (i - 1)) & 1
            sj = (state >> (j - 1)) & 1

            szsz = (0.5 - si) * (0.5 - sj)
            push!(rows, col)
            push!(cols, col)
            push!(vals, float(delta) * szsz)

            if si != sj
                flipped = state ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                if haskey(bmap, flipped)
                    push!(rows, bmap[flipped])
                    push!(cols, col)
                    push!(vals, 0.5)
                end
            end
        end
    end

    return sparse(rows, cols, vals, dim, dim)
end


function chebyshev_coefficients(target::Float64, order::Int)
    -1.0 <= target <= 1.0 || throw(ArgumentError("target must be in [-1, 1]."))
    order >= 0 || throw(ArgumentError("order must be >= 0."))
    θ = acos(target)
    coeffs = Vector{Float64}(undef, order + 1)
    for n in 0:order
        coeffs[n + 1] = (2 - (n == 0)) * cos(n * θ)
    end
    return coeffs
end


function eval_chebyshev_series_scalar(x::Real, coeffs::AbstractVector{<:Real})
    K = length(coeffs) - 1
    y = Float64(coeffs[1])
    K == 0 && return y

    tkm2 = 1.0
    tkm1 = Float64(x)
    y += Float64(coeffs[2]) * tkm1
    for k in 2:K
        tk = 2.0 * Float64(x) * tkm1 - tkm2
        y += Float64(coeffs[k + 1]) * tk
        tkm2 = tkm1
        tkm1 = tk
    end
    return y
end


function eval_chebyshev_series(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real})
    y = Vector{Float64}(undef, length(x))
    for i in eachindex(x)
        y[i] = eval_chebyshev_series_scalar(x[i], coeffs)
    end
    return y
end


function normalized_dirac_filter_coeffs(target::Float64, order::Int)
    coeffs_raw = chebyshev_coefficients(target, order)
    peak_raw = eval_chebyshev_series_scalar(target, coeffs_raw)
    abs(peak_raw) > eps(Float64) || throw(ArgumentError("Filter peak is numerically zero at target=$target."))
    coeffs_norm = coeffs_raw ./ peak_raw
    peak_norm = eval_chebyshev_series_scalar(target, coeffs_norm)
    coeffs_norm ./= peak_norm
    return coeffs_norm
end


function compute_kpm_dos_from_mapping(
    f!::Function,
    dim::Int,
    eigvals_exact::Vector{Float64};
    moments::Int,
    R::Int,
    grid_size::Int,
    rng::AbstractRNG,
)
    x0 = randn(rng, Float64, dim)
    x0 ./= norm(x0)

    mapping_cfg = Polfed.MappingConfig(Emin = first(eigvals_exact), Emax = last(eigvals_exact))
    map_plan = PolfedCore.build_mapping_plan(mapping_cfg, f!, x0, Polfed.CPU())
    fact_full = PolfedCore.FactorizationConfigFull(Polfed.FactorizationConfig(), x0, 1)
    dos_full = PolfedCore.DoSConfigFull(Polfed.DoSConfig(N = moments, R = R, kernel = :Jackson))
    PolfedCore.getdos!(dos_full, map_plan, fact_full, Polfed.CPU())

    x_grid = collect(range(-0.999, 0.999; length = grid_size))
    value_grid = map_plan.a .* x_grid .+ map_plan.b
    rho_fn = getfield(dos_full, Symbol("\u03C1"))
    dos_rescaled = rho_fn.(x_grid)
    dos_value = dos_rescaled ./ abs(map_plan.a)

    return (
        x_grid = x_grid,
        value_grid = value_grid,
        dos_rescaled = dos_rescaled,
        dos_value = dos_value,
        Emin = map_plan.Emin,
        Emax = map_plan.Emax,
        scale_a = map_plan.a,
        shift_b = map_plan.b,
    )
end


function build_transformed_mapping(
    mat::SparseMatrixCSC{Float64, Int},
    scale_a::Float64,
    shift_b::Float64,
    coeffs_norm::Vector{Float64},
)
    K = length(coeffs_norm) - 1
    buffers = Vector{Any}(undef, Threads.nthreads())
    fill!(buffers, nothing)

    h_rescaled_mul! = (Y, X) -> begin
        mul!(Y, mat, X)
        @. Y = (Y - shift_b * X) / scale_a
        nothing
    end

    function get_buffers(X)
        tid = Threads.threadid()
        buf = buffers[tid]
        T = promote_type(Float64, eltype(X))
        if buf === nothing || size(buf[1]) != size(X) || eltype(buf[1]) != T
            buf = (
                zeros(T, size(X)),
                zeros(T, size(X)),
                zeros(T, size(X)),
                zeros(T, size(X)),
            )
            buffers[tid] = buf
        end
        return buffers[tid]
    end

    return (Y, X) -> begin
        b1, b2, b3, tmp = get_buffers(X)
        fill!(b1, zero(eltype(b1)))
        fill!(b2, zero(eltype(b2)))
        fill!(b3, zero(eltype(b3)))

        for k in K:-1:1
            h_rescaled_mul!(tmp, b2)
            @. b1 = coeffs_norm[k + 1] * X + 2.0 * tmp - b3
            b1, b2, b3 = b3, b1, b2
        end

        h_rescaled_mul!(tmp, b2)
        @. Y = coeffs_norm[1] * X + tmp - b3
        nothing
    end
end


function run_spectral_tranform_data(;
    L_primary::Int = 18,
    L_secondary::AbstractVector{<:Integer} = Int[10, 12, 14],
    delta::Float64 = 1.0,
    moments_list::Vector{Int} = [25, 50, 75, 100, 125, 150, 200],
    transform_orders::Vector{Int} = [10, 20, 50, 75],
    R::Int = 500,
    target_rescaled::Float64 = 0.0,
    dos_grid_size::Int = 2001,
    seed::Int = 1234,
    output_file::String = joinpath(@__DIR__, "spectral_tranform_data_L$(L_primary).h5"),
)
    iseven(L_primary) || throw(ArgumentError("Half filling requires even L_primary. Received L_primary=$L_primary."))
    secondary_sorted = unique(sort(Int.(collect(L_secondary))))
    all(Ls -> Ls >= 2, secondary_sorted) || throw(ArgumentError("All entries in L_secondary must be >= 2."))
    all(iseven, secondary_sorted) || throw(ArgumentError("Half filling requires even values in L_secondary."))
    moments_sorted = unique(sort(moments_list))
    orders_sorted = unique(sort(transform_orders))
    all(m -> m >= 2, moments_sorted) || throw(ArgumentError("All moments must be >= 2."))
    all(k -> k >= 1, orders_sorted) || throw(ArgumentError("All transform orders must be >= 1."))

    Nup = L_primary ÷ 2
    println("Building XXZ Hamiltonian for L_primary=$L_primary, Nup=$Nup, delta=$delta")
    mat_sparse = construct_xxz_spin_sector(L_primary, delta, Nup)
    dim = size(mat_sparse, 1)
    println("Hilbert-space dimension: $dim")

    mat_dense = Matrix(mat_sparse)
    mat_dense = 0.5 .* (mat_dense .+ transpose(mat_dense))
    eigvals_main = real(eigvals(mat_dense))
    sort!(eigvals_main)

    Emin = first(eigvals_main)
    Emax = last(eigvals_main)
    scale_a = (Emax - Emin) / 2
    shift_b = (Emax + Emin) / 2
    eigvals_rescaled = (eigvals_main .- shift_b) ./ scale_a

    f_h! = (Y, X) -> mul!(Y, mat_sparse, X)

    h5open(output_file, "w") do h5
        h5["L_main"] = L_primary
        h5["L_primary"] = L_primary
        h5["L_secondary"] = secondary_sorted
        h5["delta"] = delta
        h5["R"] = R
        h5["moments_list"] = moments_sorted
        h5["transform_orders"] = orders_sorted
        h5["target_rescaled"] = target_rescaled
        h5["dos_grid_size"] = dos_grid_size
        h5["seed"] = seed

        grp_main = create_group(h5, "main")
        grp_main["L"] = L_primary
        grp_main["Nup"] = Nup
        grp_main["hilbert_dim"] = dim
        grp_main["nnz"] = nnz(mat_sparse)
        grp_main["Emin"] = Emin
        grp_main["Emax"] = Emax
        grp_main["scale_a"] = scale_a
        grp_main["shift_b"] = shift_b
        grp_main["eigvals_exact"] = eigvals_main
        grp_main["eigvals_rescaled"] = eigvals_rescaled

        grp_main_kpm = create_group(grp_main, "kpm")
        for m in moments_sorted
            println("  -> Base H KPM with moments=$m, R=$R")
            rng_m = MersenneTwister(seed + 10_000 + m)
            kpm = compute_kpm_dos_from_mapping(
                f_h!,
                dim,
                eigvals_main;
                moments = m,
                R = R,
                grid_size = dos_grid_size,
                rng = rng_m,
            )
            grp_m = create_group(grp_main_kpm, "moments_$(m)")
            grp_m["moments"] = m
            grp_m["x_grid_rescaled"] = kpm.x_grid
            grp_m["value_grid"] = kpm.value_grid
            grp_m["dos_kpm_rescaled"] = kpm.dos_rescaled
            grp_m["dos_kpm_value"] = kpm.dos_value
            grp_m["Emin"] = kpm.Emin
            grp_m["Emax"] = kpm.Emax
            grp_m["scale_a"] = kpm.scale_a
            grp_m["shift_b"] = kpm.shift_b
        end

        grp_transformed = create_group(grp_main, "transformed")
        filter_x_grid = collect(range(-0.999, 0.999; length = dos_grid_size))

        for K in orders_sorted
            println("  -> Transformed operator P0^K(H), K=$K")
            coeffs_norm = normalized_dirac_filter_coeffs(target_rescaled, K)
            filter_values = eval_chebyshev_series(filter_x_grid, coeffs_norm)
            transformed_eigs = eval_chebyshev_series(eigvals_rescaled, coeffs_norm)
            transformed_eigs_sorted = sort(copy(transformed_eigs))

            f_trans! = build_transformed_mapping(mat_sparse, scale_a, shift_b, coeffs_norm)

            grp_K = create_group(grp_transformed, "K_$(K)")
            grp_K["order"] = K
            grp_K["target_rescaled"] = target_rescaled
            grp_K["coefficients_normalized"] = coeffs_norm
            grp_K["filter_x_grid"] = filter_x_grid
            grp_K["filter_values"] = filter_values
            grp_K["transformed_eigvals"] = transformed_eigs

            grp_K_kpm = create_group(grp_K, "kpm")
            for m in moments_sorted
                println("     * KPM of transformed operator: K=$K, moments=$m, R=$R")
                rng_km = MersenneTwister(seed + 100_000 + 1000 * K + m)
                kpm_t = compute_kpm_dos_from_mapping(
                    f_trans!,
                    dim,
                    transformed_eigs_sorted;
                    moments = m,
                    R = R,
                    grid_size = dos_grid_size,
                    rng = rng_km,
                )
                grp_m = create_group(grp_K_kpm, "moments_$(m)")
                grp_m["moments"] = m
                grp_m["x_grid_rescaled"] = kpm_t.x_grid
                grp_m["value_grid"] = kpm_t.value_grid
                grp_m["dos_kpm_rescaled"] = kpm_t.dos_rescaled
                grp_m["dos_kpm_value"] = kpm_t.dos_value
                grp_m["Emin"] = kpm_t.Emin
                grp_m["Emax"] = kpm_t.Emax
                grp_m["scale_a"] = kpm_t.scale_a
                grp_m["shift_b"] = kpm_t.shift_b
            end
        end

        grp_secondary = create_group(h5, "secondary_eigenvalues")
        for Ls in secondary_sorted
            Nup_s = Ls ÷ 2
            eig_s = if Ls == L_primary
                println("  -> Reusing primary exact eigenvalues for L=$Ls")
                copy(eigvals_main)
            else
                println("  -> Computing reference eigenvalues for L=$Ls at half filling")
                mat_s = construct_xxz_spin_sector(Ls, delta, Nup_s)
                eig_tmp = real(eigvals(Matrix(mat_s)))
                sort!(eig_tmp)
                eig_tmp
            end

            grp_L = create_group(grp_secondary, "L_$(Ls)")
            grp_L["L"] = Ls
            grp_L["Nup"] = Nup_s
            grp_L["eigvals_exact"] = eig_s

            if Ls == 10
                grp_l10 = create_group(h5, "L10_reference")
                grp_l10["L"] = Ls
                grp_l10["Nup"] = Nup_s
                grp_l10["eigvals_exact"] = eig_s
            end
        end
    end

    println("Saved data to: $output_file")
    return output_file
end


run_spectral_tranform_data()
