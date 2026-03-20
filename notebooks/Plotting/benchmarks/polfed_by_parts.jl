using DrWatson
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

using CSV
using DataFrames
using PythonPlot

const XXZ_PART_ORDER = [
    "reorthogonalization",
    "QR decomposition",
    "Spectral transformation",
    "diag block",
    "convergence checking",
]

const XXZ_PART_LABELS = Dict(
    "reorthogonalization" => "Reorthogonalization",
    "QR decomposition" => "QR",
    "Spectral transformation" => "Spectral transf.",
    "diag block" => "Diag. block",
    "convergence checking" => "Convergence check",
)

const XXZ_PART_ALIASES = Dict(
    "reorthogonalization" => "reorthogonalization",
    "reorthogonlization" => "reorthogonalization",
    "reorthogonlsization" => "reorthogonalization",
    "qr decomposition" => "QR decomposition",
    "qr" => "QR decomposition",
    "spectral transformation" => "Spectral transformation",
    "spectral tranformation" => "Spectral transformation",
    "spectral transf" => "Spectral transformation",
    "diag block" => "diag block",
    "diagblock" => "diag block",
    "convergence checking" => "convergence checking",
    "convergence check" => "convergence checking",
)


function apply_xxz_profiling_style!()
    matplotlib.rcParams["figure.dpi"] = 1000
    matplotlib.rcParams["axes.grid"] = false
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.labelsize"] = 16
    matplotlib.rcParams["axes.titlesize"] = 16
    matplotlib.rcParams["xtick.labelsize"] = 13
    matplotlib.rcParams["ytick.labelsize"] = 13
    matplotlib.rcParams["legend.fontsize"] = 13
    matplotlib.rcParams["legend.title_fontsize"] = 13
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
end


function validate_columns!(df::DataFrame, required_cols::Vector{String}, csv_path::AbstractString)
    missing_cols = [col for col in required_cols if !hasproperty(df, Symbol(col))]
    isempty(missing_cols) || throw(ArgumentError("Missing required columns in $csv_path: $(join(missing_cols, ", "))"))
end


function canonical_part_name(part::AbstractString)
    key = lowercase(strip(part))
    key = replace(key, r"\s+" => " ")
    haskey(XXZ_PART_ALIASES, key) || throw(ArgumentError("Unknown part '$part'."))
    return XXZ_PART_ALIASES[key]
end


function resolve_parts_to_plot(parts_to_plot)
    parts_to_plot === nothing && return copy(XXZ_PART_ORDER)
    resolved = [canonical_part_name(String(part)) for part in parts_to_plot]
    return [part for part in XXZ_PART_ORDER if part in Set(resolved)]
end


function load_left_panel_data(csv_path::AbstractString)::DataFrame
    df = CSV.read(csv_path, DataFrame)
    validate_columns!(df, ["L", "part", "cpu_time"], csv_path)
    if hasproperty(df, :model)
        df = df[lowercase.(String.(df.model)) .== "xxz", :]
    end
    sort!(df, [:part, :L])
    return df
end


function load_right_panel_data(csv_path::AbstractString)::DataFrame
    df = CSV.read(csv_path, DataFrame)
    validate_columns!(df, ["L", "model", "part", "n_ev", "cpu_time"], csv_path)
    df = df[(lowercase.(String.(df.model)) .== "xxz") .& (Int.(df.L) .== 22), :]
    sort!(df, [:part, :n_ev])
    return df
end


function part_color_map(parts_to_plot::Vector{String})
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    return Dict(part => colors[mod1(i, length(colors))] for (i, part) in enumerate(parts_to_plot))
end


function part_linewidth()
    return 1.5
end


function part_marker()
    return "o"
end


function part_markersize()
    return 5.5
end


function plot_left_panel!(ax, df::DataFrame; parts_to_plot::Vector{String}=XXZ_PART_ORDER, color_map::Dict=part_color_map(parts_to_plot))
    for part in parts_to_plot
        subdf = df[String.(df.part) .== part, :]
        isempty(subdf) && continue

        xs = Float64.(subdf.L)
        ys = Float64.(subdf.cpu_time)
        color = color_map[part]

        ax.plot(
            xs,
            ys;
            color=color,
            linewidth=part_linewidth(),
            zorder=2,
        )
        ax.scatter(
            xs,
            ys;
            s=34,
            marker=part_marker(),
            color=color,
            edgecolors=color,
            zorder=3,
        )
    end

    ax.set_yscale("log")
    # ax.set_xlim(15.8, 22.3)
    # ax.set_xticks(collect(16:24))
    ax.set_xlabel("L")
    ax.set_ylabel("time [s]")
    ax.grid(false)
end


function plot_right_panel!(ax, df::DataFrame; parts_to_plot::Vector{String}=XXZ_PART_ORDER, color_map::Dict=part_color_map(parts_to_plot))
    for part in parts_to_plot
        subdf = df[String.(df.part) .== part, :]
        isempty(subdf) && continue

        xs = Float64.(subdf.n_ev)
        ys = Float64.(subdf.cpu_time)
        color = color_map[part]

        ax.plot(
            xs,
            ys;
            color=color,
            linewidth=part_linewidth(),
            label=XXZ_PART_LABELS[part],
            zorder=2,
        )
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-2)
    ax.set_xlabel("Nev")
    ax.set_ylabel("time [s]")
    ax.grid(false)
end


function build_right_legend(ax; parts_to_plot::Vector{String}=XXZ_PART_ORDER, color_map::Dict=part_color_map(parts_to_plot))
    line2d = matplotlib.lines.Line2D
    handles = Any[
        line2d(
            [0],
            [0];
            color=color_map[part],
            linewidth=part_linewidth(),
            marker=part_marker(),
            markerfacecolor=color_map[part],
            markeredgecolor=color_map[part],
            markersize=part_markersize(),
            label=XXZ_PART_LABELS[part],
        ) for part in parts_to_plot
    ]
    labels = [XXZ_PART_LABELS[part] for part in parts_to_plot]
    ax.legend(
        handles,
        labels;
        loc="best",
        frameon=false,
    )
end


function plot_xxz_profiling(;
    left_csv::AbstractString=joinpath(@__DIR__, "benchamrks_parts_vs_L_xxz.csv"),
    right_csv::AbstractString=joinpath(@__DIR__, "benchmarks_parts_vs_nev_L=22_xxz.csv"),
    pdf_path::AbstractString=joinpath(@__DIR__, "polfed_per_parts.pdf"),
    png_path::AbstractString=joinpath(@__DIR__, "polfed_per_parts.png"),
    figure_size::Tuple{<:Real, <:Real}=(10.5, 4.0),
    parts_to_plot=["Spectral transformation","convergence checking","reorthogonalization", "QR decomposition"],
    show_plot::Bool=true,
)
    isfile(left_csv) || throw(ArgumentError("CSV file not found: $left_csv"))
    isfile(right_csv) || throw(ArgumentError("CSV file not found: $right_csv"))

    apply_xxz_profiling_style!()
    left_df = load_left_panel_data(left_csv)
    right_df = load_right_panel_data(right_csv)
    selected_parts = resolve_parts_to_plot(parts_to_plot)
    selected_color_map = part_color_map(selected_parts)

    fig, axs = pyplot.subplots(1, 2; figsize=figure_size)
    ax_left = axs[0]
    ax_right = axs[1]

    plot_left_panel!(ax_left, left_df; parts_to_plot=selected_parts, color_map=selected_color_map)
    plot_right_panel!(ax_right, right_df; parts_to_plot=selected_parts, color_map=selected_color_map)
    build_right_legend(ax_right; parts_to_plot=selected_parts, color_map=selected_color_map)

    fig.tight_layout()
    fig.savefig(pdf_path; bbox_inches="tight")
    fig.savefig(png_path; bbox_inches="tight")

    if show_plot
        display(fig)
        fig.show()
    end

    return fig, axs, left_df, right_df
end


plot_xxz_profiling()
