using HDF5
using PythonPlot


function trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    length(x) >= 2 || return 0.0
    return sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2
end


function fit_power_law(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    mask = [x[i] > 0 && y[i] > 0 for i in eachindex(x)]
    xx = Float64[x[i] for i in eachindex(x) if mask[i]]
    yy = Float64[y[i] for i in eachindex(y) if mask[i]]
    length(xx) >= 2 || return nothing

    lx = log.(xx)
    ly = log.(yy)
    mx = sum(lx) / length(lx)
    my = sum(ly) / length(ly)
    den = sum((lx .- mx) .^ 2)
    den > 0 || return nothing
    b = sum((lx .- mx) .* (ly .- my)) / den
    a = my - b * mx
    return exp(a), b
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


function list_numeric_suffixes(grp, prefix::String)
    vals = Int[]
    rx = Regex("^" * prefix * "_(\\d+)\$")
    for key in keys(grp)
        m = match(rx, String(key))
        m === nothing && continue
        push!(vals, parse(Int, m.captures[1]))
    end
    sort!(vals)
    return vals
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


function estimate_delta_from_cutoff(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, omega::Real)
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    isempty(x) && return 0.0

    xv = Float64.(x)
    yv = Float64.(y)
    i0 = argmin(abs.(xv))
    yv[i0] < omega && return 0.0

    il = i0
    while il > 1 && yv[il - 1] >= omega
        il -= 1
    end

    ir = i0
    while ir < length(yv) && yv[ir + 1] >= omega
        ir += 1
    end

    return max(0.0, (xv[ir] - xv[il]) / 2)
end


function setup_matplotlib!(;
    use_tex::Bool = true,
    figure_dpi::Int = 160,
    save_dpi::Int = 300,
    apply_style::Bool = true,
    style_name::String = "rok-custom",
)
    pyplot = PythonPlot.pyplot
    mpl = PythonPlot.matplotlib

    apply_style && pyplot.style.use([style_name])
    mpl.rcParams["figure.dpi"] = figure_dpi
    mpl.rcParams["savefig.dpi"] = save_dpi
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.top"] = true
    mpl.rcParams["ytick.right"] = true

    if use_tex
        mpl.rcParams["text.usetex"] = true
        mpl.rcParams["text.latex.preamble"] = raw"\usepackage{amsmath}"
    else
        mpl.rcParams["text.usetex"] = false
    end
end


function density_panel!(
    ax,
    rawvals::AbstractVector{<:Real};
    color,
    xlabel::AbstractString,
    panel_label::AbstractString,
    ylim,
    xlim,
    nbins::Int,
    line_y::Union{Nothing, AbstractVector{<:Real}} = nothing,
    line_x::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    h = ax.hist(
        rawvals;
        bins = nbins,
        density = true,
        orientation = "horizontal",
        color = color,
        alpha = 0.90,
        edgecolor = "black",
        linewidth = 0.5,
    )

    if !isnothing(line_y) && !isnothing(line_x)
        ax.plot(line_x, line_y, color = "black", linewidth = 1.6)
    else
        counts = Float64.(collect(h[1]))
        edges = Float64.(collect(h[2]))
        centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
        ax.plot(counts, centers, color = "black", linewidth = 1.4)
    end

    # Handle Julia tuples/vectors and Python tuples uniformly.
    _limits2(v, name::AbstractString) = begin
        vals = collect(v)
        length(vals) >= 2 || throw(ArgumentError("$name must contain at least two values."))
        return vals[1], vals[2]
    end

    y0, y1 = _limits2(ylim, "ylim")
    x0, x1 = _limits2(xlim, "xlim")

    ax.set_xlabel(xlabel)
    ax.set_ylim(y0, y1)
    # ax.set_xlim(x0, x1)
    ax.text(0.70, 0.78, panel_label, transform = ax.transAxes, fontsize = 14)
    ax.tick_params(direction = "in")
end


function plot_spectral_transformation_figure(;
    filepath::String = joinpath(@__DIR__, "xxz_spectral_transform_L14.h5"),
    orders::NTuple{3, Int} = (10, 20, 50),
    omega::Float64 = 0.17,
    delta_for_arrow::Union{Nothing, Float64} = nothing,
    apply_style::Bool = true,
    style_name::String = "rok-custom",
    use_tex::Bool = true,
    save_png::Union{Nothing, String} = nothing,
    save_pdf::Union{Nothing, String} = nothing,
    show_plot::Bool = true,
    moments_for_panel_a::Union{Nothing, Integer, AbstractVector{<:Integer}} = nothing,
    show_panel_a_legend::Bool = true,
    Ks_inset::AbstractVector{<:Integer} = Int[10, 20, 50, 100, 200],
    inset_target::Float64 = 0.0,
    save_inset_values_path::Union{Nothing, String} = joinpath(@__DIR__, "inset_counts.txt"),
    eps_xlim::Tuple{Float64, Float64} = (-1.1, 0.75),
    right_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    dens_xlim::Tuple{Float64, Float64} = (0.0, 3.0),
    bins_main::Int = 45,
    bins_transformed::NTuple{3, Int} = (45, 45, 45),
    fig_size::Tuple{Float64, Float64} = (10.0, 4.3),
    show_transformed_scatter::Bool = true,
    scatter_size::Real = 8,
    scatter_alpha::Real = 0.45,
    normalize_kpm_untransformed::Bool = true,
)
    setup_matplotlib!(use_tex = use_tex, apply_style = apply_style, style_name = style_name)
    mpl = PythonPlot.matplotlib
    pyplot = PythonPlot.pyplot

    plotting_colors = collect(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    ncolors = length(plotting_colors)
    ncolors > 0 || throw(ArgumentError("Style color cycle is empty."))
    pick_color(i::Int) = plotting_colors[((i - 1) % ncolors) + 1]

    colA = pick_color(1)
    col10 = pick_color(2)
    col20 = pick_color(3)
    col50 = pick_color(4)
    order_colors = Dict(orders[1] => col10, orders[2] => col20, orders[3] => col50)

    exact_centers = Float64[]
    exact_density = Float64[]
    exact_eigvals_rescaled = Float64[]
    panel_a_moments = Int[]
    panel_a_curves = Dict{Int, NamedTuple{(:x, :rho), Tuple{Vector{Float64}, Vector{Float64}}}}()

    order_data = Dict{Int, NamedTuple}()
    all_orders = Int[]

    h5open(filepath, "r") do h5
        haskey(h5, "exact") || throw(ArgumentError("Missing group: exact"))
        haskey(h5, "kpm") || throw(ArgumentError("Missing group: kpm"))
        haskey(h5, "chebyshev_delta") || throw(ArgumentError("Missing group: chebyshev_delta"))

        exact_centers = Vector{Float64}(read(h5["exact/dos_hist_centers"]))
        exact_density = Vector{Float64}(read(h5["exact/dos_hist_density"]))
        exact_eigvals_rescaled = Vector{Float64}(read(h5["exact/eigvals_rescaled"]))

        available_moments = list_numeric_suffixes(h5["kpm"], "moments")
        isempty(available_moments) && throw(ArgumentError("No moments_* groups found in kpm."))
        panel_a_moments = resolve_requested_moments(moments_for_panel_a, available_moments)
        for m in panel_a_moments
            grp = "kpm/moments_$(m)"
            panel_a_curves[m] = (
                x = Vector{Float64}(read(h5["$grp/x_grid_rescaled"])),
                rho = Vector{Float64}(read(h5["$grp/dos_kpm_rescaled"])),
            )
        end

        all_orders = list_numeric_suffixes(h5["chebyshev_delta"], "order")
        all(o -> o in all_orders, orders) || throw(ArgumentError("Requested orders=$orders not all present. Available: $all_orders"))

        for o in orders
            base = "chebyshev_delta/order_$(o)"
            order_data[o] = (
                filter_x = Vector{Float64}(read(h5["$base/filter_x_grid"])),
                filter_y = Vector{Float64}(read(h5["$base/filter_values"])),
                eig_tr = Vector{Float64}(read(h5["$base/transformed_eigvals"])),
                hist_centers = Vector{Float64}(read(h5["$base/transformed_dos_hist_centers"])),
                hist_density = Vector{Float64}(read(h5["$base/transformed_dos_hist_density"])),
            )
        end

    end

    Ks_inset_list = Int.(collect(Ks_inset))
    isempty(Ks_inset_list) && throw(ArgumentError("Ks_inset cannot be empty."))

    K_list = Int[]
    Nev_list = Int[]
    for K in Ks_inset_list
        coeffs_norm = normalized_dirac_filter_coeffs(inset_target, K)
        transformed_vals = eval_chebyshev_series(exact_eigvals_rescaled, coeffs_norm)
        nev = count(v -> v >= omega, transformed_vals)
        push!(K_list, K)
        push!(Nev_list, nev)
    end

    if !isnothing(save_inset_values_path)
        mkpath(dirname(save_inset_values_path))
        open(save_inset_values_path, "w") do io
            println(io, "# K Nev_above_omega")
            for i in eachindex(K_list)
                println(io, K_list[i], " ", Nev_list[i])
            end
        end
    end

    delta_use = isnothing(delta_for_arrow) ?
        estimate_delta_from_cutoff(order_data[orders[end]].filter_x, order_data[orders[end]].filter_y, omega) :
        delta_for_arrow

    panel_a_curves_plot = Dict{Int, NamedTuple{(:x, :rho), Tuple{Vector{Float64}, Vector{Float64}}}}()
    for m in panel_a_moments
        x = panel_a_curves[m].x
        rho = copy(panel_a_curves[m].rho)
        if normalize_kpm_untransformed
            integral = trapz(x, rho)
            if integral > 0 && isfinite(integral)
                rho ./= integral
            end
        end
        panel_a_curves_plot[m] = (x = x, rho = rho)
    end

    fig = pyplot.figure(figsize = fig_size)

    # Two explicit sub-layouts avoid Python slice objects and keep the same geometry.
    gs_left = fig.add_gridspec(
        2,
        1,
        left = 0.06,
        right = 0.66,
        bottom = 0.12,
        top = 0.95,
        height_ratios = [1.0, 2.2],
        hspace = 0.12,
    )
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_b = fig.add_subplot(gs_left[1, 0], sharex = ax_a)

    # Right panels are aligned to panel (b) only, so they look like projections.
    bbox_b = ax_b.get_position()
    gs_right = fig.add_gridspec(
        1,
        3,
        left = 0.70,
        right = 0.98,
        bottom = bbox_b.y0,
        top = bbox_b.y1,
        wspace = 0.18,
    )

    ax_c = fig.add_subplot(gs_right[0, 0])
    ax_d = fig.add_subplot(gs_right[0, 1], sharey = ax_c)
    ax_e = fig.add_subplot(gs_right[0, 2], sharey = ax_c)
    ax_a.tick_params(labelbottom = false)

    h_a = ax_a.hist(
        exact_eigvals_rescaled;
        bins = bins_main,
        density = true,
        color = colA,
        alpha = 0.85,
        edgecolor = "black",
        linewidth = 0.5,
    )
    ax_a.plot(exact_centers, exact_density, color = "black", linewidth = 1.2, alpha = 0.9)
    single_panel_a = length(panel_a_moments) == 1
    for (i, m) in enumerate(panel_a_moments)
        curve_color = single_panel_a ? "black" : pick_color(i + 1)
        ax_a.plot(
            panel_a_curves_plot[m].x,
            panel_a_curves_plot[m].rho,
            color = curve_color,
            linewidth = 1.6,
            label = "\$K=$(m)\$",
        )
    end
    if show_panel_a_legend && !single_panel_a
        ax_a.legend(loc = "upper right", frameon = false)
    end
    ax_a.set_ylabel(raw"$\rho(\varepsilon)$")
    ax_a.set_xlim(eps_xlim...)
    ax_a.text(0.03, 0.78, raw"$(a)$", transform = ax_a.transAxes, fontsize = 14)

    for o in orders
        ax_b.plot(
            order_data[o].filter_x,
            order_data[o].filter_y,
            color = order_colors[o],
            linewidth = 2.0,
            linestyle = "-",
            label = "\$K=$(o)\$",
        )

        if show_transformed_scatter
            ax_b.scatter(
                exact_eigvals_rescaled,
                order_data[o].eig_tr;
                s = scatter_size,
                color = order_colors[o],
                alpha = scatter_alpha,
                linewidths = 0.0,
                zorder = 6,
            )
        end
    end

    ax_b.axhline(omega, color = "black", linewidth = 1.6)

    if delta_use > 0
        y_arrow = omega - 0.10
        arrow_color = order_colors[orders[end]]
        ax_b.annotate(
            "",
            xy = (+delta_use, y_arrow),
            xytext = (-delta_use, y_arrow),
            arrowprops = Dict("arrowstyle" => "<->", "color" => arrow_color, "lw" => 2.0),
        )
        ax_b.text(0.0, y_arrow - 0.07, raw"$2\delta$", color = arrow_color, ha = "center", va = "top")
    end

    ax_b.legend(loc = "upper right", frameon = false)
    ax_b.set_xlabel(raw"$\varepsilon$")
    ax_b.set_ylabel(raw"$P_0^{K}(\varepsilon)$")
    ax_b.set_xlim(eps_xlim...)
    ax_b.text(0.03, 0.08, raw"$(b)$", transform = ax_b.transAxes, fontsize = 14)

    right_ylim_use = if isnothing(right_ylim)
        ax_b.get_ylim()
    else
        right_ylim
    end

    valid_idx = [i for i in eachindex(K_list) if K_list[i] > 0 && Nev_list[i] > 0]
    K_plot = Float64[K_list[i] for i in valid_idx]
    Nev_plot = Float64[Nev_list[i] for i in valid_idx]

    if !isempty(K_plot)
        ax_in = ax_b.inset_axes([0.10, 0.58, 0.33, 0.35])
        ax_in.scatter(K_plot, Nev_plot; color = col20, s = 18)
        fit = fit_power_law(K_plot, Nev_plot)
        if !isnothing(fit)
            A, b = fit
            Kfine = collect(range(minimum(K_plot), maximum(K_plot); length = 200))
            ax_in.plot(Kfine, A .* (Kfine .^ b), "k--", linewidth = 1.5)
            ax_in.text(0.46, 0.74, "\$\\propto K^{$(round(b; digits=2))}\$", transform = ax_in.transAxes)
        end
        ax_in.set_xscale("log")
        ax_in.set_yscale("log")
        ax_in.set_xlabel(raw"$K$", labelpad = -2)
        ax_in.set_ylabel(raw"$N_{\mathrm{ev}}$", labelpad = -4)
        ax_in.tick_params(which = "both", direction = "in", pad = 1)
        ax_in.tick_params(which = "both", direction = "in", top = true, right = true)
    end

    for ax in (ax_c, ax_d, ax_e)
        ax.tick_params(labelleft = false)
    end

    density_panel!(
        ax_c,
        order_data[orders[1]].eig_tr;
        color = order_colors[orders[1]],
        xlabel = "\$\\rho\\!\\left(P_0^{$(orders[1])}(\\varepsilon)\\right)\$",
        panel_label = raw"$(c)$",
        ylim = right_ylim_use,
        xlim = dens_xlim,
        nbins = bins_transformed[1],
        line_y = order_data[orders[1]].hist_centers,
        line_x = order_data[orders[1]].hist_density,
    )

    density_panel!(
        ax_d,
        order_data[orders[2]].eig_tr;
        color = order_colors[orders[2]],
        xlabel = "\$\\rho\\!\\left(P_0^{$(orders[2])}(\\varepsilon)\\right)\$",
        panel_label = raw"$(d)$",
        ylim = right_ylim_use,
        xlim = dens_xlim,
        nbins = bins_transformed[2],
        line_y = order_data[orders[2]].hist_centers,
        line_x = order_data[orders[2]].hist_density,
    )

    density_panel!(
        ax_e,
        order_data[orders[3]].eig_tr;
        color = order_colors[orders[3]],
        xlabel = "\$\\rho\\!\\left(P_0^{$(orders[3])}(\\varepsilon)\\right)\$",
        panel_label = raw"$(e)$",
        ylim = right_ylim_use,
        xlim = dens_xlim,
        nbins = bins_transformed[3],
        line_y = order_data[orders[3]].hist_centers,
        line_x = order_data[orders[3]].hist_density,
    )

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e)
        ax.tick_params(which = "both", direction = "in", top = true, right = true)
    end

    if !isnothing(save_png)
        mkpath(dirname(save_png))
        fig.savefig(save_png, bbox_inches = "tight")
    end
    if !isnothing(save_pdf)
        mkpath(dirname(save_pdf))
        fig.savefig(save_pdf, bbox_inches = "tight")
    end
    show_plot && display(fig)
    return fig, (ax_a, ax_b, ax_c, ax_d, ax_e)
end

plot_spectral_transformation_figure()
