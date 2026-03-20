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


function normalized_histogram(
    values::AbstractVector{<:Real},
    nbins::Int;
    left::Float64 = minimum(Float64.(values)),
    right::Float64 = maximum(Float64.(values)),
)
    nbins > 0 || throw(ArgumentError("nbins must be positive."))
    right > left || throw(ArgumentError("right must be larger than left."))

    edges = collect(range(left, right; length = nbins + 1))
    counts = zeros(Float64, nbins)
    width = (right - left) / nbins

    for v in values
        x = Float64(v)
        if x < left || x > right
            continue
        end
        idx = min(nbins, Int(floor((x - left) / width)) + 1)
        counts[idx] += 1.0
    end

    density = counts ./ (length(values) * width)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return centers, density, width
end


function compute_kpm_dos(
    mat::SparseMatrixCSC{Float64, Int},
    eigvals_exact::Vector{Float64};
    moments::Int,
    R::Int,
    grid_size::Int,
    rng::AbstractRNG,
)
    dim = size(mat, 1)
    x0 = randn(rng, Float64, dim)
    x0 ./= norm(x0)

    f! = (Y, X) -> mul!(Y, mat, X)
    mapping_cfg = Polfed.MappingConfig(Emin = first(eigvals_exact), Emax = last(eigvals_exact))
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


function eval_chebyshev_series(x::AbstractVector{<:Real}, coeffs::AbstractVector{<:Real})
    K = length(coeffs) - 1
    xv = Float64.(x)
    y = fill(Float64(coeffs[1]), length(xv))
    K == 0 && return y

    Tkm2 = ones(Float64, length(xv))
    Tkm1 = copy(xv)
    y .+= Float64(coeffs[2]) .* Tkm1

    for k in 2:K
        Tk = @. 2.0 * xv * Tkm1 - Tkm2
        y .+= Float64(coeffs[k + 1]) .* Tk
        Tkm2 = Tkm1
        Tkm1 = Tk
    end
    return y
end


function eval_chebyshev_series_scalar(x::Real, coeffs::AbstractVector{<:Real})
    K = length(coeffs) - 1
    y = Float64(coeffs[1])
    K == 0 && return y

    Tkm2 = 1.0
    Tkm1 = Float64(x)
    y += Float64(coeffs[2]) * Tkm1

    for k in 2:K
        Tk = 2.0 * Float64(x) * Tkm1 - Tkm2
        y += Float64(coeffs[k + 1]) * Tk
        Tkm2 = Tkm1
        Tkm1 = Tk
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

    return coeffs_norm, peak_raw
end


function run_xxz_spectral_transform_scan(;
    L::Int = 14,
    delta::Float64 = 1.0,
    R::Int = 500,
    kpm_moments::Vector{Int} = [25, 50, 100, 200],
    transformed_dos_reference_moment::Union{Nothing, Int} = nothing,
    transform_orders::Vector{Int} = [10, 20, 50],
    dos_grid_size::Int = 2001,
    hist_bins::Int = 220,
    seed::Int = 1234,
    output_file::String = joinpath(@__DIR__, "xxz_spectral_transform_L$(L).h5"),
)
    moments_list = unique(sort(kpm_moments))
    orders_list = unique(sort(transform_orders))
    all(m -> m >= 2, moments_list) || throw(ArgumentError("All KPM moments must be >= 2."))
    all(k -> k >= 1, orders_list) || throw(ArgumentError("All transform orders must be >= 1."))
    dos_grid_size >= 101 || throw(ArgumentError("dos_grid_size must be >= 101."))
    hist_bins >= 20 || throw(ArgumentError("hist_bins must be >= 20."))

    transformed_ref_moment = isnothing(transformed_dos_reference_moment) ?
        first(moments_list) :
        transformed_dos_reference_moment
    transformed_ref_moment in moments_list || throw(
        ArgumentError(
            "transformed_dos_reference_moment=$transformed_ref_moment must be one of kpm_moments=$moments_list",
        ),
    )

    iseven(L) || throw(ArgumentError("Half filling requires even L. Received L=$L."))
    Nup = L ÷ 2
    println("Building XXZ model at half filling: L=$L, Nup=$Nup, delta=$delta")
    mat_sparse = construct_xxz_spin_sector(L, delta, Nup)
    dim = size(mat_sparse, 1)
    println("Hilbert-space dimension: $dim")

    mat = Matrix(mat_sparse)
    mat = 0.5 .* (mat .+ transpose(mat))
    eigvals_exact = real(eigvals(mat))
    sort!(eigvals_exact)

    Emin = first(eigvals_exact)
    Emax = last(eigvals_exact)
    spread = Emax - Emin
    spread > 0 || throw(ArgumentError("Degenerate spectrum: Emax == Emin."))
    scale_a = spread / 2
    shift_b = (Emax + Emin) / 2
    eigvals_rescaled = (eigvals_exact .- shift_b) ./ scale_a

    hist_centers_exact, hist_density_exact, hist_width_exact =
        normalized_histogram(eigvals_rescaled, hist_bins; left = -1.0, right = 1.0)

    x_filter_grid = collect(range(-0.999, 0.999; length = dos_grid_size))

    h5open(output_file, "w") do h5
        h5["L"] = L
        h5["Nup"] = Nup
        h5["delta"] = delta
        h5["R"] = R
        h5["seed"] = seed
        h5["kpm_moments"] = moments_list
        h5["transformed_dos_reference_moment"] = transformed_ref_moment
        h5["transform_orders"] = orders_list
        h5["dos_grid_size"] = dos_grid_size
        h5["hist_bins"] = hist_bins

        grp_exact = create_group(h5, "exact")
        grp_exact["hilbert_dim"] = dim
        grp_exact["nnz"] = nnz(mat_sparse)
        grp_exact["Emin"] = Emin
        grp_exact["Emax"] = Emax
        grp_exact["scale_a"] = scale_a
        grp_exact["shift_b"] = shift_b
        grp_exact["eigvals_exact"] = eigvals_exact
        grp_exact["eigvals_rescaled"] = eigvals_rescaled
        grp_exact["dos_hist_centers"] = hist_centers_exact
        grp_exact["dos_hist_density"] = hist_density_exact
        grp_exact["dos_hist_bin_width"] = hist_width_exact

        grp_kpm = create_group(h5, "kpm")
        kpm_reference_x = nothing
        kpm_reference_rho = nothing
        for m in moments_list
            println("  -> KPM DoS with moments=$m and R=$R")
            kpm_rng = MersenneTwister(seed + 10_000 + m)
            kpm = compute_kpm_dos(
                mat_sparse,
                eigvals_exact;
                moments = m,
                R = R,
                grid_size = dos_grid_size,
                rng = kpm_rng,
            )

            grp_m = create_group(grp_kpm, "moments_$(m)")
            grp_m["N_moments"] = m
            grp_m["x_grid_rescaled"] = kpm.x_grid
            grp_m["energy_grid"] = kpm.energy_grid
            grp_m["dos_kpm_rescaled"] = kpm.dos_rescaled
            grp_m["dos_kpm_energy"] = kpm.dos_energy
            grp_m["Emin"] = kpm.Emin
            grp_m["Emax"] = kpm.Emax
            grp_m["scale_a"] = kpm.scale_a
            grp_m["shift_b"] = kpm.shift_b

            if m == transformed_ref_moment
                kpm_reference_x = kpm.x_grid
                kpm_reference_rho = kpm.dos_rescaled
            end
        end

        grp_transform = create_group(h5, "chebyshev_delta")
        target = 0.0
        for order in orders_list
            coeffs_norm, raw_peak = normalized_dirac_filter_coeffs(target, order)
            filter_grid = eval_chebyshev_series(x_filter_grid, coeffs_norm)
            filter_at_target = eval_chebyshev_series_scalar(target, coeffs_norm)
            transformed_eigvals = eval_chebyshev_series(eigvals_rescaled, coeffs_norm)

            left_t = minimum(transformed_eigvals)
            right_t = maximum(transformed_eigvals)
            if !(right_t > left_t)
                left_t -= 1e-12
                right_t += 1e-12
            end
            centers_t, density_t, width_t = normalized_histogram(
                transformed_eigvals,
                hist_bins;
                left = left_t,
                right = right_t,
            )

            filtered_exact_dos = hist_density_exact .* eval_chebyshev_series(hist_centers_exact, coeffs_norm)
            filtered_kpm_ref = Float64[]
            if !isnothing(kpm_reference_x) && !isnothing(kpm_reference_rho)
                filtered_kpm_ref = kpm_reference_rho .* eval_chebyshev_series(kpm_reference_x, coeffs_norm)
            end

            grp_o = create_group(grp_transform, "order_$(order)")
            grp_o["order"] = order
            grp_o["target_rescaled"] = target
            grp_o["coefficients_normalized"] = coeffs_norm
            grp_o["raw_peak_before_normalization"] = raw_peak
            grp_o["normalized_peak_at_target"] = filter_at_target
            grp_o["filter_x_grid"] = x_filter_grid
            grp_o["filter_values"] = filter_grid
            grp_o["transformed_eigvals"] = transformed_eigvals
            grp_o["transformed_dos_hist_centers"] = centers_t
            grp_o["transformed_dos_hist_density"] = density_t
            grp_o["transformed_dos_hist_bin_width"] = width_t
            grp_o["filtered_exact_dos_centers"] = hist_centers_exact
            grp_o["filtered_exact_dos"] = filtered_exact_dos
            if !isempty(filtered_kpm_ref)
                grp_o["filtered_kpm_reference_x"] = kpm_reference_x
                grp_o["filtered_kpm_reference_dos"] = filtered_kpm_ref
                grp_o["filtered_kpm_reference_moments"] = transformed_ref_moment
            end

            println(
                "  -> Transform order=$order: normalized peak at target=0 is ",
                round(filter_at_target; digits = 12),
            )
        end
    end

    println("Saved data to: $output_file")
    return output_file
end


if abspath(PROGRAM_FILE) == @__FILE__
    L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 14
    delta = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0

    run_xxz_spectral_transform_scan(;
        L = L,
        delta = delta,
        R = 500,
        kpm_moments = [25, 50, 100, 200],
        transformed_dos_reference_moment = 50,
        transform_orders = [10, 20, 50],
    )
end
