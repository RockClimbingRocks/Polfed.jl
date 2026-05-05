using HDF5
using PythonPlot


function trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    length(x) >= 2 || throw(ArgumentError("Need at least 2 points for trapezoidal integration."))
    return sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2
end


gaussian_pdf(x::Real, μ::Real, σ::Real) = exp(-0.5 * ((x - μ) / σ)^2) / (σ * sqrt(2π))


function linear_interp_sorted(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, xq::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    length(x) >= 2 || throw(ArgumentError("Need at least 2 points for interpolation."))

    xv = Float64.(x)
    yv = Float64.(y)
    out = Vector{Float64}(undef, length(xq))

    j = 1
    for (i, xqi) in enumerate(xq)
        xqf = Float64(xqi)
        if xqf <= xv[1]
            out[i] = yv[1]
            continue
        elseif xqf >= xv[end]
            out[i] = yv[end]
            continue
        end

        while j < length(xv) - 1 && xv[j + 1] < xqf
            j += 1
        end
        x1, x2 = xv[j], xv[j + 1]
        y1, y2 = yv[j], yv[j + 1]
        t = (xqf - x1) / (x2 - x1)
        out[i] = y1 + t * (y2 - y1)
    end
    return out
end


function normalized_histogram(values::AbstractVector{<:Real}, nbins::Int; left::Float64=-1.0, right::Float64=1.0)
    nbins > 0 || throw(ArgumentError("nbins must be positive."))
    right > left || throw(ArgumentError("right must be larger than left."))

    edges = collect(range(left, right; length=nbins + 1))
    counts = zeros(Float64, nbins)
    width = (right - left) / nbins

    for v in values
        if v < left || v > right
            continue
        end
        idx = min(nbins, Int(floor((v - left) / width)) + 1)
        counts[idx] += 1.0
    end

    density = counts ./ (length(values) * width)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    return centers, density, width
end


function list_available_moments(h5::HDF5.File, N::Int)
    group_name = "N_$(N)"
    haskey(h5, group_name) || throw(ArgumentError("Group '$group_name' not found in HDF5 file."))
    grp = h5[group_name]

    moments = Int[]
    for key in keys(grp)
        m = match(r"^moments_(\d+)$", String(key))
        m === nothing && continue
        push!(moments, parse(Int, m.captures[1]))
    end
    sort!(moments)
    return moments
end


function validate_requested_moments(requested::AbstractVector{<:Integer}, available::AbstractVector{<:Integer})
    req = Int[]
    seen = Set{Int}()
    for m in requested
        mm = Int(m)
        mm in seen && continue
        push!(req, mm)
        push!(seen, mm)
    end
    missing = [m for m in req if !(m in available)]
    isempty(missing) || throw(ArgumentError("Requested moments not found: $missing; available: $available"))
    return req
end


resolve_requested_moments(::Nothing, available::AbstractVector{<:Integer}) = Int.(available)
resolve_requested_moments(requested::Integer, available::AbstractVector{<:Integer}) = validate_requested_moments(Int[requested], available)
function resolve_requested_moments(requested::AbstractVector{<:Integer}, available::AbstractVector{<:Integer})
    isempty(requested) && return Int.(available)
    return validate_requested_moments(requested, available)
end


function plot_dos_vs_moments(;
    filepath::String = joinpath(@__DIR__, "disordered_j1j2_dos_L20_N5.h5"),
    N::Int = 5,
    moments_orders::Union{Nothing, Integer, AbstractVector{<:Integer}} = 100,
    nbins::Int = 25,
    normalize_kpm::Bool = true,
    plot_gaussian::Bool = true,
    apply_style::Bool = true,
    style_name::String = "rok-custom",
    dpi::Int = 1000,
    axis_label_fontsize::Real = 9,
    main_xlim::Union{Nothing, Tuple{<:Real, <:Real}} = (-1.0, 1.5),
    inset_rect::NTuple{4, Float64} = (0.6, 0.55, 0.375, 0.40),
    savepath::Union{Nothing, String} = "/home/rokpintar/projects/Polfed/plots/dos/dos-new.pdf",
    savepath_png::Union{Nothing, String} = "/home/rokpintar/projects/Polfed/plots/dos/dos-new.png",
    png_dpi::Int = 1000,
    show_plot::Bool = true,
)
    matplotlib = PythonPlot.matplotlib
    pyplot = PythonPlot.pyplot

    eigvals_exact = Float64[]
    selected_moments = Int[]
    curves_x = Dict{Int, Vector{Float64}}()
    curves_rho = Dict{Int, Vector{Float64}}()

    h5open(filepath, "r") do h5
        available = list_available_moments(h5, N)
        selected_moments = resolve_requested_moments(moments_orders, available)

        base = "N_$(N)"
        eigvals_exact = Vector{Float64}(read(h5["$base/eigvals_exact"]))

        for m in selected_moments
            grp = "$base/moments_$(m)"
            x = Vector{Float64}(read(h5["$grp/x_grid_rescaled"]))
            rho = Vector{Float64}(read(h5["$grp/dos_kpm_rescaled"]))
            curves_x[m] = x
            curves_rho[m] = rho
        end
    end

    Emin = minimum(eigvals_exact)
    Emax = maximum(eigvals_exact)
    spread = Emax - Emin
    spread > 0 || throw(ArgumentError("Exact eigenvalues are degenerate (Emax == Emin)."))
    eps_exact = @. 2.0 * (eigvals_exact - Emin) / spread - 1.0
    hist_centers, hist_density, hist_width = normalized_histogram(eps_exact, nbins; left=-1.0, right=1.0)
    μ_exact = sum(eps_exact) / length(eps_exact)
    σ_exact = sqrt(sum((eps_exact .- μ_exact) .^ 2) / length(eps_exact))

    if apply_style
        pyplot.style.use([style_name])
        matplotlib.rcParams["figure.dpi"] = dpi
    end
    plotting_colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    ncolors = length(plotting_colors)

    fig, ax = pyplot.subplots()
    hist_color = plotting_colors[0]
    kpm_color = plotting_colors[1]
    gauss_color = plotting_colors[2]
    main_legend_handles = Any[]
    main_legend_labels = String[]
    ax.bar(
        hist_centers,
        hist_density;
        width=hist_width,
        align="center",
        color=hist_color,
        alpha=0.75,
        edgecolor=hist_color,
        linewidth=0.4,
        label=raw"$\mathrm{Exact}$",
    )
    ax.scatter(
        hist_centers,
        hist_density;
        s=1.75,
        color="black",
        zorder=6,
    )

    x_gauss = Float64[]
    y_gauss = Float64[]
    if plot_gaussian && isfinite(σ_exact) && σ_exact > 0
        x_gauss = collect(range(-1.0, 1.0; length=1000))
        y_gauss = gaussian_pdf.(x_gauss, μ_exact, σ_exact)
        gauss_handle = ax.plot(
            x_gauss,
            y_gauss;
            color=gauss_color,
            linewidth=1.0,
            linestyle="--",
        )
        push!(main_legend_handles, gauss_handle[0])
        push!(main_legend_labels, raw"$\mathrm{Gaussian}$")
    end

    kpm_ref_x = Float64[]
    kpm_ref_rho = Float64[]
    for (i, m) in enumerate(selected_moments)
        x = curves_x[m]
        rho = copy(curves_rho[m])
        if normalize_kpm
            integral = trapz(x, rho)
            if integral > 0 && isfinite(integral)
                rho ./= integral
            end
        end
        if i == 1
            kpm_ref_x = copy(x)
            kpm_ref_rho = copy(rho)
        end
        color = i == 1 ? kpm_color : plotting_colors[mod(i, ncolors)]
        kpm_handle = ax.plot(x, rho; color=color, linewidth=1.0)
        if i == 1
            pushfirst!(main_legend_handles, kpm_handle[0])
            pushfirst!(main_legend_labels, raw"$\mathrm{KPM}$")
        end
    end

    ax.set_xlabel(L"\tilde{E}", fontsize=axis_label_fontsize)
    ax.set_ylabel(L"\rho(\tilde{E})", fontsize=axis_label_fontsize)
    if !isnothing(main_xlim)
        ax.set_xlim(main_xlim...)
    end
    ax.set_ylim(bottom=0.0)

    if !isempty(kpm_ref_x)
        ax_in = ax.inset_axes(collect(inset_rect))
        kpm_interp = linear_interp_sorted(kpm_ref_x, kpm_ref_rho, hist_centers)
        kpm_err = abs.(kpm_interp .- hist_density)
        ax_in.plot(hist_centers, kpm_err; color=kpm_color, linewidth=0.75, marker="o", markersize=1.5, label=raw"$\mathrm{KPM}$")

        if plot_gaussian && !isempty(x_gauss)
            gauss_interp = linear_interp_sorted(x_gauss, y_gauss, hist_centers)
            gauss_err = abs.(gauss_interp .- hist_density)
            ax_in.plot(hist_centers, gauss_err; color=gauss_color, linewidth=0.75, linestyle="--", marker="s", markersize=1.4, label=raw"$\mathrm{Gaussian}$")
        end

        ax_in.set_yscale("log")
        ax_in.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_in.set_xlabel(L"\tilde{E}", fontsize=5, labelpad=1)
        ax_in.set_ylabel("error", fontsize=5, labelpad=1)
        ax_in.tick_params(axis="both", labelsize=5, direction="in")
    end

    if !isempty(main_legend_handles)
        ax.legend(main_legend_handles, main_legend_labels, frameon=false, fontsize=6, loc="upper left")
    end
    fig.tight_layout()

    if !isnothing(savepath)
        mkpath(dirname(savepath))
        fig.savefig(savepath, bbox_inches="tight")
        println("Saved plot: ", abspath(savepath))
    end
    if !isnothing(savepath_png)
        mkpath(dirname(savepath_png))
        fig.savefig(savepath_png, dpi=png_dpi, bbox_inches="tight")
    end

    show_plot && display(fig)
    return fig, ax
end


plot_dos_vs_moments()
