using DrWatson
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

using CSV
using DataFrames
using PythonPlot

const INPUT_CSV = joinpath(@__DIR__, "benchmarks_nev_cpu_by_L.csv")
const OUTPUT_BASENAME = joinpath(@__DIR__, "benchmarks_nev_cpu_by_L")
const L_VALUES = [14, 16, 18, 20, 22]

const LINE_STYLES = Dict(
    22 => "-",
    20 => "--",
    18 => "-.",
    16 => ":",
    14 => (0, (5, 1.5)),
)


function apply_publication_style!(; use_custom_style::Bool=true)
    if use_custom_style
        try
            pyplot.style.use(["rok-custom"])
        catch
        end
    end
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.grid"] = false
    matplotlib.rcParams["axes.labelsize"] = 20
    matplotlib.rcParams["axes.titlesize"] = 20
    matplotlib.rcParams["xtick.labelsize"] = 16
    matplotlib.rcParams["ytick.labelsize"] = 16
    matplotlib.rcParams["legend.fontsize"] = 14
    matplotlib.rcParams["legend.title_fontsize"] = 16
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.minor.visible"] = true
    matplotlib.rcParams["ytick.minor.visible"] = true
end


function build_line_color_map(L_values::Vector{Int})
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    ordered_L = sort(unique(L_values))
    return Dict(L => colors[(i - 1) % length(colors)] for (i, L) in enumerate(ordered_L))
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


function plot_panel!(ax, df::DataFrame, ycol::Symbol, ylabel::AbstractString, L_values::Vector{Int}; color_map::Dict)
    for L in L_values
        subdf = df[Int.(df.L) .== L, :]
        isempty(subdf) && continue

        ax.plot(
            Float64.(subdf.n_ev),
            Float64.(subdf[!, ycol]);
            color=color_map[L],
            linestyle=LINE_STYLES[L],
            linewidth=2.4,
            label="L = $(L)",
        )
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(raw"$N_{\mathrm{ev}}$")
    ax.set_ylabel(ylabel)
    ax.grid(false)
end


function build_right_legend!(ax, L_values::Vector{Int}; color_map::Dict)
    line2d = matplotlib.lines.Line2D
    handles = Any[
        line2d(
            [0],
            [0];
            color=color_map[L],
            linestyle=LINE_STYLES[L],
            linewidth=2.4,
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


function add_panel_label!(ax, label::AbstractString)
    ax.text(
        0.96,
        0.06,
        label;
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=20,
    )
end


function plot_nev_cpu_by_L(;
    input_csv::AbstractString=INPUT_CSV,
    output_basename::AbstractString=OUTPUT_BASENAME,
    L_values::Vector{Int}=copy(L_VALUES),
    figure_size::Tuple{<:Real, <:Real}=(12.5, 4.0),
    use_custom_style::Bool=true,
    show_plot::Bool=true,
)
    isfile(input_csv) || throw(ArgumentError("CSV file not found: $input_csv"))

    apply_publication_style!(; use_custom_style=use_custom_style)
    color_map = build_line_color_map(L_values)
    df = load_data(input_csv, L_values)

    fig, axs = pyplot.subplots(1, 2; figsize=figure_size)
    ax_left = axs[0]
    ax_right = axs[1]

    plot_panel!(ax_left, df, :cpu_time, raw"$t_{\mathrm{CPU}}\,[\mathrm{s}]$", L_values; color_map=color_map)
    plot_panel!(ax_right, df, :cpu_time_per_nev, raw"$t_{\mathrm{CPU}} / N_{\mathrm{ev}}\,[\mathrm{s}]$", L_values; color_map=color_map)
    add_panel_label!(ax_left, "(a)")
    add_panel_label!(ax_right, "(b)")
    build_right_legend!(ax_right, L_values; color_map=color_map)

    fig.subplots_adjust(wspace=0.16)
    fig.tight_layout()
    pdf_path = "$(output_basename).pdf"
    png_path = "$(output_basename).png"
    fig.savefig(pdf_path; bbox_inches="tight")
    println("Saved PDF: ", abspath(pdf_path))
    fig.savefig(png_path; bbox_inches="tight")
    println("Saved PNG: ", abspath(png_path))

    if show_plot
        display(fig)
        fig.show()
    end

end


plot_nev_cpu_by_L()
