using HDF5
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
    moments_orders::Union{Nothing, Integer, AbstractVector{<:Integer}} = nothing,
    nbins::Int = 45,
    normalize_kpm::Bool = true,
    apply_style::Bool = true,
    style_name::String = "rok-custom",
    dpi::Int = 1000,
    savepath::Union{Nothing, String} = "/home/rokpintar/projects/Polfed/plots/dos/dos_moments.pdf",
    savepath_png::Union{Nothing, String} = "/home/rokpintar/projects/Polfed/plots/dos/dos_moments.png",
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

    if apply_style
        pyplot.style.use([style_name])
        matplotlib.rcParams["figure.dpi"] = dpi
    end
    plotting_colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = pyplot.subplots()
    hist_color = plotting_colors[0]
    ax.bar(
        hist_centers,
        hist_density;
        width=hist_width,
        align="center",
        color=hist_color,
        alpha=0.75,
        edgecolor=hist_color,
        linewidth=0.4,
    )

    ncolors = length(plotting_colors)
    for (i, m) in enumerate(selected_moments)
        x = curves_x[m]
        rho = copy(curves_rho[m])
        if normalize_kpm
            integral = trapz(x, rho)
            if integral > 0 && isfinite(integral)
                rho ./= integral
            end
        end
        color = plotting_colors[mod(i, ncolors)]
        ax.plot(x, rho; color=color, linewidth=1.0)
    end

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(Float64[])
    ax.set_yticks(Float64[])
    fig.tight_layout()

    if !isnothing(savepath)
        mkpath(dirname(savepath))
        fig.savefig(savepath, bbox_inches="tight")
    end
    if !isnothing(savepath_png)
        mkpath(dirname(savepath_png))
        fig.savefig(savepath_png, dpi=png_dpi, bbox_inches="tight")
    end

    show_plot && display(fig)
    return fig, ax
end


plot_dos_vs_moments()
