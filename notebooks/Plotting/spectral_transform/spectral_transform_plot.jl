using HDF5
using PythonPlot
using Polynomials


function trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    length(x) >= 2 || return 0.0
    return sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2
end


function fit_inverse_with_offset(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    mask = [x[i] > 0 && isfinite(y[i]) for i in eachindex(x)]
    xx = Float64[x[i] for i in eachindex(x) if mask[i]]
    yy = Float64[y[i] for i in eachindex(y) if mask[i]]
    length(xx) >= 2 || return nothing

    invx = 1.0 ./ xx
    mx = sum(invx) / length(invx)
    my = sum(yy) / length(yy)
    den = sum((invx .- mx) .^ 2)
    den > 0 || return nothing
    a = sum((invx .- mx) .* (yy .- my)) / den
    b = my - a * mx
    return a, b
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


function threshold_crossings_around_zero(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, omega::Real)
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    isempty(x) && return nothing

    xv = Float64.(x)
    yv = Float64.(y)
    p = sortperm(xv)
    xv = xv[p]
    yv = yv[p]
    n = length(xv)

    i0 = argmin(abs.(xv))
    yv[i0] >= omega || return nothing

    il = i0
    while il > 1 && yv[il - 1] >= omega
        il -= 1
    end

    ir = i0
    while ir < n && yv[ir + 1] >= omega
        ir += 1
    end

    interp_x(x1, y1, x2, y2, yt) = abs(y2 - y1) <= eps(Float64) ? (x1 + x2) / 2 : x1 + (yt - y1) * (x2 - x1) / (y2 - y1)
    x_left = il == 1 ? xv[1] : interp_x(xv[il - 1], yv[il - 1], xv[il], yv[il], omega)
    x_right = ir == n ? xv[n] : interp_x(xv[ir], yv[ir], xv[ir + 1], yv[ir + 1], omega)

    return x_left, x_right
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
    mpl.rcParams["axes.linewidth"] = 0.75
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
        ax.plot(line_x, line_y, color = "black", linewidth = 1.25)
    else
        counts = Float64.(collect(h[1]))
        edges = Float64.(collect(h[2]))
        centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
        ax.plot(counts, centers, color = "black", linewidth = 1.25)
    end

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
    ax.set_xscale("log")
    ax.text(0.70, 0.78, panel_label, transform = ax.transAxes, fontsize = 14)
    ax.tick_params(direction = "in")
end


function plot_spectral_transform_figure(;
    filepath::String = joinpath(@__DIR__, "spectral_tranform_data_L18.h5"),
    orders::NTuple{3, Int} = (10, 20, 50),
    omega::Float64 = 0.17,
    delta_for_arrow::Union{Nothing, Float64} = nothing,
    apply_style::Bool = true,
    style_name::String = "rok-custom",
    use_tex::Bool = true,
    save_png::Union{Nothing, String} = nothing,
    save_pdf::Union{Nothing, String} = nothing,
    show_plot::Bool = true,
    kpm_moment_main::Int = 100,
    Ks_inset::AbstractVector{<:Integer} = Int.(collect(LinRange(10,1000,100)).÷1),
    inset_fit_K_range::Tuple{Float64, Float64} = (100.0, 1000.0),
    inset_plot_K_range::Tuple{Float64, Float64} = (10.0, 1000.0),
    inset_rect::NTuple{4, Float64} = (0.025, 0.55, 0.4, 0.4),
    inset_xlabel_coords::Tuple{Float64, Float64} = (0.5, -0.2),
    inset_ylabel_coords::Tuple{Float64, Float64} = (0.125, 0.5),
    inset_target::Float64 = 0.,
    save_inset_values_path::Union{Nothing, String} = joinpath(@__DIR__, "inset_counts_new_data.txt"),
    eps_xlim::Tuple{Float64, Float64} = (-1., 0.75),
    right_ylim::Union{Nothing, Tuple{Float64, Float64}} = nothing,
    dens_xlim::Tuple{Float64, Float64} = (0.0, 3.0),
    transformed_kpm_value_range::Tuple{Float64, Float64} = (-0.21, 0.99),
    bins_main::Int = 40,
    moments_transformed::NTuple{3, Int} = (100, 100, 100),
    bins_transformed::NTuple{3, Int} = (35, 35, 35),
    fig_size::Tuple{Float64, Float64} = (10.0, 4.3),
    width_ab::Float64 = 0.40,
    width_cde::Float64 = 0.1,
    panel_spacing_ab::Float64 = 0.025,
    panel_spacing_right::Float64 = 0.01,
    panel_spacing::Union{Nothing, Float64} = nothing,
    scatter_L::Union{Nothing, Int} = 12,
    scatter_L_size::Real = 8,
    scatter_L_sizes_by_order::NTuple{3, Float64} = (1.0, 0.875, 0.75),
    scatter_L_alpha::Real = 0.95,
    normalize_kpm_untransformed::Bool = true,
    normalize_kpm_transformed::Bool = true,
)
    setup_matplotlib!(use_tex = use_tex, apply_style = apply_style, style_name = style_name)
    mpl = PythonPlot.matplotlib
    pyplot = PythonPlot.pyplot

    plotting_colors = collect(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    ncolors = length(plotting_colors)
    ncolors > 0 || throw(ArgumentError("Style color cycle is empty."))
    pick_color(i::Int) = plotting_colors[((i - 1) % ncolors) + 1]

    colA = pick_color(5)
    col10 = pick_color(1)
    col20 = pick_color(2)
    col50 = pick_color(3)
    order_colors = Dict(orders[1] => col10, orders[2] => col20, orders[3] => col50)

    main_eigvals_rescaled = Float64[]
    scatter_eigvals_rescaled = Float64[]
    main_kpm_x = Float64[]
    main_kpm_rho = Float64[]
    order_data = Dict{Int, NamedTuple}()

    h5open(filepath, "r") do h5
        haskey(h5, "main") || throw(ArgumentError("Missing group: main"))
        haskey(h5, "main/kpm") || throw(ArgumentError("Missing group: main/kpm"))
        haskey(h5, "main/transformed") || throw(ArgumentError("Missing group: main/transformed"))

        moments_available = Int.(read(h5["moments_list"]))
        kpm_moment_main in moments_available || throw(ArgumentError("kpm_moment_main=$kpm_moment_main not in moments_list=$moments_available"))
        all(m -> m in moments_available, moments_transformed) || throw(ArgumentError("moments_transformed=$moments_transformed not all in moments_list=$moments_available"))
        order_moment = Dict(orders[i] => moments_transformed[i] for i in eachindex(orders))

        main_eigvals_rescaled = Vector{Float64}(read(h5["main/eigvals_rescaled"]))
        main_kpm_x = Vector{Float64}(read(h5["main/kpm/moments_$(kpm_moment_main)/x_grid_rescaled"]))
        main_kpm_rho = Vector{Float64}(read(h5["main/kpm/moments_$(kpm_moment_main)/dos_kpm_rescaled"]))
        if !isnothing(scatter_L)
            L_scatter_val = Int(scatter_L)
            eigvals_scatter_exact = nothing
            key_secondary = "secondary_eigenvalues/L_$(L_scatter_val)/eigvals_exact"

            if haskey(h5, key_secondary)
                eigvals_scatter_exact = Vector{Float64}(read(h5[key_secondary]))
            else
                available = haskey(h5, "L_secondary") ? Int.(read(h5["L_secondary"])) : Int[]
                throw(ArgumentError("Requested scatter_L=$L_scatter_val not found in file. Available L_secondary=$(available)."))
            end

            if !isempty(eigvals_scatter_exact)
                emin_s = minimum(eigvals_scatter_exact)
                emax_s = maximum(eigvals_scatter_exact)
                if emax_s > emin_s
                    a_s = (emax_s - emin_s) / 2
                    b_s = (emax_s + emin_s) / 2
                    scatter_eigvals_rescaled = (eigvals_scatter_exact .- b_s) ./ a_s
                else
                    scatter_eigvals_rescaled = zeros(length(eigvals_scatter_exact))
                end
            end
        end

        for o in orders
            base = "main/transformed/K_$(o)"
            haskey(h5, base) || throw(ArgumentError("Missing transformed order group: $base"))
            mt = order_moment[o]
            kpm_value_grid = Vector{Float64}(read(h5["$base/kpm/moments_$(mt)/value_grid"]))
            kpm_dos_value = Vector{Float64}(read(h5["$base/kpm/moments_$(mt)/dos_kpm_value"]))
            if normalize_kpm_transformed
                integral_t = trapz(kpm_value_grid, kpm_dos_value)
                if integral_t > 0 && isfinite(integral_t)
                    kpm_dos_value ./= integral_t
                end
            end

            vals = Vector{Float64}(read(h5["/main/eigvals_rescaled"]))
            # coefficients_normalize = Vector{Float64}(read(h5["$base/coefficients_normalized"]))

            # target = 0.
            # K =100
            # coefs = [(2 - (n == 0)) * cos(n * acos(target)) for n in 0:K]
            # p = ChebyshevT(coefs)
            # p2(x) = p(x)/p(target)
            # vals_transform = p2.(vals)

            # println(length)


            order_data[o] = (
                filter_x = Vector{Float64}(read(h5["$base/filter_x_grid"])),
                filter_y = Vector{Float64}(read(h5["$base/filter_values"])),
                coeffs_norm = Vector{Float64}(read(h5["$base/coefficients_normalized"])),
                eig_tr = Vector{Float64}(read(h5["$base/transformed_eigvals"])),
                # eig_tr = vals_transform,
                kpm_value_grid = kpm_value_grid,
                kpm_dos_value = kpm_dos_value,
            )
        end
    end

    Ks_inset_list = Int.(collect(Ks_inset))
    isempty(Ks_inset_list) && throw(ArgumentError("Ks_inset cannot be empty."))
    K_list = Int[]
    Nev_list = Int[]
    for K in Ks_inset_list
        coeffs_norm = normalized_dirac_filter_coeffs(inset_target, K)
        transformed_vals = eval_chebyshev_series(main_eigvals_rescaled, coeffs_norm)
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

    largest_order = orders[end]
    crossings = threshold_crossings_around_zero(order_data[largest_order].filter_x, order_data[largest_order].filter_y, omega)
    if isnothing(crossings)
        delta_fallback = isnothing(delta_for_arrow) ?
            estimate_delta_from_cutoff(order_data[largest_order].filter_x, order_data[largest_order].filter_y, omega) :
            delta_for_arrow
        x_left = -delta_fallback
        x_right = +delta_fallback
    else
        x_left, x_right = crossings
        if !isnothing(delta_for_arrow)
            x_left = -delta_for_arrow
            x_right = +delta_for_arrow
        end
    end

    main_kpm_plot = copy(main_kpm_rho)
    if normalize_kpm_untransformed
        integral = trapz(main_kpm_x, main_kpm_plot)
        if integral > 0 && isfinite(integral)
            main_kpm_plot ./= integral
        end
    end

    fig = pyplot.figure(figsize = fig_size)
    left_margin = 0.06
    right_margin = 0.02
    bottom_margin = 0.12
    top_margin = 0.95
    hratio_top = 1.0
    hratio_bottom = 2.2

    spacing_ab = isnothing(panel_spacing) ? panel_spacing_ab : panel_spacing
    spacing_right = isnothing(panel_spacing) ? panel_spacing_right : panel_spacing

    spacing_ab > 0 || throw(ArgumentError("panel_spacing_ab must be > 0, got $spacing_ab"))
    spacing_right > 0 || throw(ArgumentError("panel_spacing_right must be > 0, got $spacing_right"))
    width_ab > 0 || throw(ArgumentError("width_ab must be > 0, got $width_ab"))
    width_cde > 0 || throw(ArgumentError("width_cde must be > 0, got $width_cde"))

    x_c = left_margin + width_ab + spacing_right
    x_d = x_c + width_cde + spacing_right
    x_e = x_d + width_cde + spacing_right
    x_end = x_e + width_cde
    x_end <= 1.0 - right_margin || throw(ArgumentError("Layout does not fit. Decrease width_ab/width_cde/panel_spacing_right or increase figure width."))

    total_h = top_margin - bottom_margin
    total_h > spacing_ab || throw(ArgumentError("Vertical space is too small for panel_spacing_ab=$spacing_ab."))
    h_available = total_h - spacing_ab
    hsum = hratio_top + hratio_bottom
    h_a = h_available * hratio_top / hsum
    h_b = h_available * hratio_bottom / hsum
    y_b = bottom_margin
    y_a = y_b + h_b + spacing_ab

    ax_b = fig.add_axes([left_margin, y_b, width_ab, h_b])
    ax_a = fig.add_axes([left_margin, y_a, width_ab, h_a], sharex = ax_b)
    ax_c = fig.add_axes([x_c, y_b, width_cde, h_b])
    ax_d = fig.add_axes([x_d, y_b, width_cde, h_b], sharey = ax_c)
    ax_e = fig.add_axes([x_e, y_b, width_cde, h_b], sharey = ax_c)
    ax_a.tick_params(labelbottom = false)

    ax_a.hist(
        main_eigvals_rescaled;
        bins = bins_main,
        density = true,
        color = colA,
        alpha = 0.85,
        edgecolor = "black",
        linewidth = 0.5,
    )
    ax_a.plot(main_kpm_x, main_kpm_plot, color = "black", linewidth = 1.6)
    ax_a.set_ylabel(raw"$\rho(\varepsilon)$")
    ax_a.set_xlim(eps_xlim...)
    ax_a.text(0.03, 0.78, raw"$(a)$", transform = ax_a.transAxes, fontsize = 14)

    linestyle_by_order = Dict(10 => "-", 20 => "-", 50 => "-")
    marker_by_order = Dict(10 => "o", 20 => "s", 50 => "v")
    for (i, o) in enumerate(orders)
        s_ref = scatter_L_size * scatter_L_sizes_by_order[i]
        ax_b.plot(
            order_data[o].filter_x,
            order_data[o].filter_y,
            color = order_colors[o],
            linewidth = 1.5,
            linestyle = get(linestyle_by_order, o, "-"),
            label = "\$K=$(o)\$",
        )
        if !isempty(scatter_eigvals_rescaled)
            scatter_tr = eval_chebyshev_series(scatter_eigvals_rescaled, order_data[o].coeffs_norm)
            ax_b.scatter(
                scatter_eigvals_rescaled,
                scatter_tr;
                s = s_ref,
                marker = get(marker_by_order, o, "o"),
                facecolors = order_colors[o],
                edgecolors = order_colors[o],
                alpha = scatter_L_alpha,
                linewidths = 0.9,
                zorder = 7,
            )
        end
    end

    ax_b.axhline(omega, color = "black", linewidth = 1.6, label = raw"$\Omega$")
    if x_right > x_left
        y_dashed_bottom = -0.25
        y_arrow = -0.25
        arrow_color = order_colors[largest_order]
        ax_b.vlines([x_left, x_right], y_dashed_bottom, omega, colors = "black", linestyles = "--", linewidth = 1.2)
        ax_b.annotate(
            "",
            xy = (x_right, y_arrow),
            xytext = (x_left, y_arrow),
            arrowprops = Dict("arrowstyle" => "<->", "color" => arrow_color, "lw" => 1.3, "shrinkA" => 0, "shrinkB" => 0),
        )
        ax_b.text((x_left + x_right) / 2, y_arrow + 0.02, raw"$2\delta$", color = arrow_color, ha = "center", va = "bottom")

    end

    ax_b.legend(loc = "upper right", frameon = false)
    ax_b.set_xlabel(raw"$\varepsilon$", fontsize = 18)
    ax_b.set_ylabel(raw"$P_0^{K}(\varepsilon)$")
    ax_b.set_xlim(eps_xlim...)
    ax_b.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    ax_b.text(0.85, 0.5, raw"$(b)$", transform = ax_b.transAxes, fontsize = 14)

    right_ylim_use = isnothing(right_ylim) ? ax_b.get_ylim() : right_ylim
    kpm_min, kpm_max = transformed_kpm_value_range
    kpm_max > kpm_min || throw(ArgumentError("transformed_kpm_value_range must satisfy min < max. Got $transformed_kpm_value_range"))

    clip_kpm_curve(y::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = begin
        mask = (y .>= kpm_min) .& (y .<= kpm_max)
        return y[mask], x[mask]
    end

    line_y_c, line_x_c = clip_kpm_curve(order_data[orders[1]].kpm_value_grid, order_data[orders[1]].kpm_dos_value)
    line_y_d, line_x_d = clip_kpm_curve(order_data[orders[2]].kpm_value_grid, order_data[orders[2]].kpm_dos_value)
    line_y_e, line_x_e = clip_kpm_curve(order_data[orders[3]].kpm_value_grid, order_data[orders[3]].kpm_dos_value)

    valid_idx = [i for i in eachindex(K_list) if K_list[i] > 0 && Nev_list[i] > 0]
    K_plot = Float64[K_list[i] for i in valid_idx]
    Nev_plot = Float64[Nev_list[i] for i in valid_idx]
    fit_kmin, fit_kmax = inset_fit_K_range
    plot_kmin, plot_kmax = inset_plot_K_range
    inset_x0, inset_y0, inset_w, inset_h = inset_rect
    xlabel_x, xlabel_y = inset_xlabel_coords
    ylabel_x, ylabel_y = inset_ylabel_coords
    fit_kmax > fit_kmin || throw(ArgumentError("inset_fit_K_range must satisfy min < max. Got $inset_fit_K_range"))
    plot_kmax > plot_kmin || throw(ArgumentError("inset_plot_K_range must satisfy min < max. Got $inset_plot_K_range"))
    plot_kmin > 0 || throw(ArgumentError("inset_plot_K_range must be positive for log-x inset. Got $inset_plot_K_range"))
    inset_w > 0 || throw(ArgumentError("inset_rect width must be > 0. Got $inset_w"))
    inset_h > 0 || throw(ArgumentError("inset_rect height must be > 0. Got $inset_h"))
    if !isempty(K_plot)
        ax_in = ax_b.inset_axes([inset_x0, inset_y0, inset_w, inset_h])
        ax_in.scatter(K_plot, Nev_plot; color = pick_color(4), s = 18)
        fit_idx = [i for i in eachindex(K_plot) if fit_kmin <= K_plot[i] <= fit_kmax]
        K_fit = Float64[K_plot[i] for i in fit_idx]
        Nev_fit = Float64[Nev_plot[i] for i in fit_idx]
        fit = fit_inverse_with_offset(K_fit, Nev_fit)
        if !isnothing(fit)
            a, b = fit
            Kfine = collect(range(plot_kmin, plot_kmax; length = 200))
            yfit = a ./ Kfine .+ b
            pos = yfit .> 0
            if any(pos)
                ax_in.plot(Kfine[pos], yfit[pos], linestyle = "--", color = "black", linewidth = 1.5)
                ax_in.text(0.54, 0.78, raw"$\propto 1/K$", transform = ax_in.transAxes)
            end
        end
        ax_in.set_xscale("log")
        ax_in.set_yscale("log")
        ax_in.set_xlabel(raw"$K$", labelpad = 2)
        ax_in.xaxis.set_label_coords(xlabel_x, xlabel_y)
        ax_in.set_ylabel(raw"$N_{\mathrm{ev}}$", labelpad = -4)
        ax_in.yaxis.set_label_coords(ylabel_x, ylabel_y)
        ax_in.tick_params(axis = "x", which = "both", direction = "in", pad = 4)
        ax_in.tick_params(axis = "y", which = "both", direction = "in", pad = 1)
        ax_in.tick_params(axis = "y", which = "both", labelleft = false, labelright = true)
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
        line_y = line_y_c,
        line_x = line_x_c,
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
        line_y = line_y_d,
        line_x = line_x_d,
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
        line_y = line_y_e,
        line_x = line_x_e,
    )

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e)
        ax.tick_params(which = "both", direction = "in", top = true, right = true)
    end

    if !isnothing(save_png)
        mkpath(dirname(save_png))
        fig.savefig(save_png, bbox_inches = "tight")
        println("Saved plot: ", abspath(save_png))
    end
    if !isnothing(save_pdf)
        mkpath(dirname(save_pdf))
        fig.savefig(save_pdf, bbox_inches = "tight")
        println("Saved plot: ", abspath(save_pdf))
    end
    show_plot && display(fig)
    return fig, (ax_a, ax_b, ax_c, ax_d, ax_e)
end


plot_spectral_transform_figure(
    save_pdf = joinpath(@__DIR__, "..", "..", "..", "plots", "SpectralTransformationOfEigenvalues.pdf"),
)
