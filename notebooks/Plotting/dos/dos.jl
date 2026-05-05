using HDF5
using LaTeXStrings
using PythonPlot


function trapz(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length."))
    length(x) >= 2 || throw(ArgumentError("Need at least 2 points for trapezoidal integration."))
    return sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2
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

    norm = length(values) * width
    density = counts ./ norm
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    return edges, density, centers, width
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


function maxdos_target(x_rescaled::AbstractVector{<:Real}, rho_rescaled::AbstractVector{<:Real})
    length(x_rescaled) == length(rho_rescaled) || throw(ArgumentError("x_rescaled and rho_rescaled must have the same length."))
    !isempty(x_rescaled) || throw(ArgumentError("KPM arrays are empty."))
    return Float64(x_rescaled[argmax(rho_rescaled)])
end


function target_to_rescaled(target_spec, eps_maxdos::Float64, scale_a::Float64, shift_b::Float64)
    if target_spec isa Symbol
        target_spec === :maxdos && return eps_maxdos
        target_spec === :middle && return 0.0
        throw(ArgumentError("Unknown target symbol: $target_spec"))
    end

    if target_spec isa Tuple && length(target_spec) == 2 && target_spec[1] isa Symbol
        tag = target_spec[1]
        val = Float64(target_spec[2])

        if tag === :offset
            -1.0 <= val <= 1.0 || throw(ArgumentError("Offset target requires eta in [-1, 1], got $val"))
            return val >= 0 ? eps_maxdos + val * (1.0 - eps_maxdos) : eps_maxdos + val * (1.0 + eps_maxdos)
        end
        tag === :unrescaled && return (val - shift_b) / scale_a
        tag === :rescaled && return val
        throw(ArgumentError("Unknown target tuple tag: $tag"))
    end

    if target_spec isa Real
        return (Float64(target_spec) - shift_b) / scale_a
    end

    throw(ArgumentError("Unsupported target specification: $target_spec"))
end


function plot_rescaled_dos_with_kpm(;
    filepath::String = joinpath(@__DIR__, "disordered_j1j2_dos_L20_N5.h5"),
    N::Int = 5,
    moments::Int = 100,
    nbins::Int = 45,
    normalize_kpm::Bool = true,
    apply_style::Bool = false,
    style_name::String = "rok-custom",
    dpi::Int = 1000,
    targets::AbstractVector = Any[
        :maxdos,
        :middle,
        (:offset, -0.5),
        (:offset, 0.5),
        (:unrescaled, 0.),
        (:rescaled, 0.5),
    ],
    savepath::Union{Nothing, String} = nothing,
    show_plot::Bool = true,
)
    matplotlib = PythonPlot.matplotlib
    pyplot = PythonPlot.pyplot

    eigvals_exact = Float64[]
    x_kpm = Float64[]
    rho_kpm = Float64[]
    moments_available = Int[]
    scale_a = NaN
    shift_b = NaN

    h5open(filepath, "r") do h5
        moments_available = list_available_moments(h5, N)
        moments in moments_available || throw(
            ArgumentError("moments=$moments not found. Available moments for N=$N: $moments_available")
        )

        base = "N_$(N)"
        kpm_group = "N_$(N)/moments_$(moments)"
        eigvals_exact = Vector{Float64}(read(h5["$base/eigvals_exact"]))
        x_kpm = Vector{Float64}(read(h5["$kpm_group/x_grid_rescaled"]))
        rho_kpm = Vector{Float64}(read(h5["$kpm_group/dos_kpm_rescaled"]))
        if haskey(h5, "$kpm_group/scale_a")
            scale_a = Float64(read(h5["$kpm_group/scale_a"]))
        end
        if haskey(h5, "$kpm_group/shift_b")
            shift_b = Float64(read(h5["$kpm_group/shift_b"]))
        end
    end

    Emin = minimum(eigvals_exact)
    Emax = maximum(eigvals_exact)
    spread = Emax - Emin
    spread > 0 || throw(ArgumentError("Exact eigenvalues are degenerate (Emax == Emin)."))
    if !isfinite(scale_a) || !isfinite(shift_b) || scale_a == 0.0
        scale_a = spread / 2
        shift_b = (Emax + Emin) / 2
    end

    eps_exact = @. 2.0 * (eigvals_exact - Emin) / spread - 1.0
    eps_maxdos = maxdos_target(x_kpm, rho_kpm)

    _, rho_hist, centers, bin_width = normalized_histogram(eps_exact, nbins; left=-1.0, right=1.0)

    rho_kpm_plot = copy(rho_kpm)
    if normalize_kpm
        integral = trapz(x_kpm, rho_kpm_plot)
        if integral > 0 && isfinite(integral)
            rho_kpm_plot ./= integral
        end
    end

    if apply_style
        pyplot.style.use([style_name])
        matplotlib.rcParams["figure.dpi"] = dpi
    end
    plotting_colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = pyplot.subplots()

    hist_color = plotting_colors[0]
    kpm_color = plotting_colors[1]

    # hist_label = raw"$$"
    # bins_label = raw"$\mathrm{Bin\ centers}$"
    kpm_label = "\$\\mathrm{KPM}\$"

    ax.bar(
        centers,
        rho_hist;
        width=bin_width,
        align="center",
        color=hist_color,
        alpha=0.75,
        # edgecolor="black",
        edgecolor=hist_color,
        linewidth=0.5,
        # label=hist_label,
    )
    ax.scatter(
        centers,
        rho_hist;
        s=1.75,
        color=hist_color,
        zorder=5,
    )

    ax.plot(
        x_kpm,
        rho_kpm_plot;
        color=kpm_color,
        linewidth=1.,
        label=kpm_label,
    )

    lss = Any[
        "-",                 # solid
        "--",                # dashed
        "-.",                # dash-dot
        ":",                 # dotted
        (0, (7, 2)),         # long dash
        (0, (3, 1, 1, 1)),   # dash-dot-dot
    ]
    nlss = length(lss)
    for (i, t) in enumerate(targets)
        eps_t = target_to_rescaled(t, eps_maxdos, scale_a, shift_b)
        ls_t = lss[mod1(i, nlss)]
        label_t = "$(sprint(show, t))"
        ax.axvline(
            eps_t;
            color="black",
            linestyle=ls_t,
            linewidth=0.5,
            alpha=0.999,
            label=label_t,
        )
    end

    ax.set_xlabel(L"\tilde{E}", fontsize=9)
    ax.set_ylabel(L"\rho(\tilde{E})", fontsize=9)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(bottom=0.0)
    # ax.grid(false, alpha=0.25)
    ax.legend(frameon=false, fontsize=4)
    fig.tight_layout()

    if !isnothing(savepath)
        fig.savefig(savepath, bbox_inches="tight")
    end

    show_plot && display(fig)
    return fig, ax
end


plot_rescaled_dos_with_kpm(; apply_style=true)
