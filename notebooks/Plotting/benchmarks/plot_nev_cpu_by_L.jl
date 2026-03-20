using DrWatson
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

using CSV
using DataFrames
using PythonPlot

const INPUT_CSV = joinpath(@__DIR__, "benchmarks_nev_cpu_by_L.csv")
const OUTPUT_BASENAME = joinpath(@__DIR__, "benchmarks_nev_cpu_by_L")
const L_VALUES = [14, 16, 18, 20, 22]

const LINE_COLORS = Dict(
    14 => "tab:blue",
    16 => "tab:orange",
    18 => "tab:green",
    20 => "tab:red",
    22 => "tab:purple",
)

const LINE_STYLES = Dict(
    14 => "-",
    16 => "--",
    18 => "-.",
    20 => ":",
    22 => (0, (5, 1.5)),
)


function apply_publication_style!()
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.grid"] = false
    matplotlib.rcParams["axes.labelsize"] = 12
    matplotlib.rcParams["axes.titlesize"] = 12
    matplotlib.rcParams["xtick.labelsize"] = 10
    matplotlib.rcParams["ytick.labelsize"] = 10
    matplotlib.rcParams["legend.fontsize"] = 10
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.minor.visible"] = true
    matplotlib.rcParams["ytick.minor.visible"] = true
end


function validate_columns!(df::DataFrame, required_cols::Vector{String}, csv_path::AbstractString)
    missing_cols = [col for col in required_cols if !hasproperty(df, Symbol(col))]
    isempty(missing_cols) || throw(ArgumentError("Missing required columns in $csv_path: $(join(missing_cols, ", "))"))
end


function load_data(csv_path::AbstractString, L_values::Vector{Int})::DataFrame
    df = CSV.read(csv_path, DataFrame)
    validate_columns!(df, ["L", "n_ev", "cpu_time", "cpu_time_per_nev"], csv_path)
    df = df[in.(Int.(df.L), Ref(L_values)), :]
    sort!(df, [:L, :n_ev])
    return df
end


function plot_panel!(ax, df::DataFrame, ycol::Symbol, ylabel::AbstractString, L_values::Vector{Int})
    for L in L_values
        subdf = df[Int.(df.L) .== L, :]
        isempty(subdf) && continue

        ax.plot(
            Float64.(subdf.n_ev),
            Float64.(subdf[!, ycol]);
            color=LINE_COLORS[L],
            linestyle=LINE_STYLES[L],
            linewidth=1.8,
            label="L = $(L)",
        )
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(raw"$N_{\mathrm{ev}}$")
    ax.set_ylabel(ylabel)
    ax.grid(false)
end


function build_right_legend!(ax, L_values::Vector{Int})
    line2d = matplotlib.lines.Line2D
    handles = Any[
        line2d(
            [0],
            [0];
            color=LINE_COLORS[L],
            linestyle=LINE_STYLES[L],
            linewidth=1.8,
            label="L = $(L)",
        ) for L in L_values
    ]
    labels = ["L = $(L)" for L in L_values]
    ax.legend(
        handles,
        labels;
        loc="best",
        frameon=false,
    )
end


function plot_nev_cpu_by_L(;
    input_csv::AbstractString=INPUT_CSV,
    output_basename::AbstractString=OUTPUT_BASENAME,
    L_values::Vector{Int}=copy(L_VALUES),
    figure_size::Tuple{<:Real, <:Real}=(10.5, 4.0),
    show_plot::Bool=true,
)
    isfile(input_csv) || throw(ArgumentError("CSV file not found: $input_csv"))

    apply_publication_style!()
    df = load_data(input_csv, L_values)

    fig, axs = pyplot.subplots(1, 2; figsize=figure_size)
    ax_left = axs[0]
    ax_right = axs[1]

    plot_panel!(ax_left, df, :cpu_time, raw"$t_{\mathrm{cpu}}$", L_values)
    plot_panel!(ax_right, df, :cpu_time_per_nev, raw"$t_{\mathrm{cpu}} / N_{\mathrm{ev}}$", L_values)
    build_right_legend!(ax_right, L_values)

    fig.subplots_adjust(wspace=0.16)
    fig.tight_layout()
    fig.savefig("$(output_basename).pdf"; bbox_inches="tight")
    fig.savefig("$(output_basename).png"; bbox_inches="tight")

    if show_plot
        display(fig)
        fig.show()
    end

    return fig, axs, df
end


plot_nev_cpu_by_L()
