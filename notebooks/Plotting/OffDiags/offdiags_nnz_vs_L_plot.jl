using DrWatson
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

using CSV
using DataFrames
using Printf
using PythonPlot

const INPUT_CSV = joinpath(@__DIR__, "offdiags_nnz_vs_L.csv")
const OUTPUT_BASENAME = joinpath(@__DIR__, "Number_offdiagonals_per_row")
const MODEL_ORDER = ["XXZ", "SYK4d2", "J_1-J_2", "J_1-J_2-J_3", "SYK4d3"]
const MODEL_LABELS = Dict(
    "XXZ" => raw"$\mathrm{XXZ}$",
    "SYK4d2" => raw"$\mathrm{SYK}_4\mathrm{d}2$",
    "J_1-J_2" => raw"$J_1$-$J_2$",
    "J_1-J_2-J_3" => raw"$J_1$-$J_2$-$J_3$",
    "SYK4d3" => raw"$\mathrm{SYK}_4\mathrm{d}3$",
)
const MODEL_MARKERS = Dict(
    "XXZ" => "o",
    "SYK4d2" => "*",
    "SYK4d3" => "s",
    "J_1-J_2" => "^",
    "J_1-J_2-J_3" => "D",
)


function apply_offdiags_style!(; use_custom_style::Bool=true)
    if use_custom_style
        try
            pyplot.style.use(["rok-custom"])
        catch
        end
    end
    matplotlib.rcParams["figure.dpi"] = 1000
    matplotlib.rcParams["savefig.dpi"] = 1000
    matplotlib.rcParams["axes.grid"] = false
    matplotlib.rcParams["axes.labelsize"] = 24
    matplotlib.rcParams["xtick.labelsize"] = 14
    matplotlib.rcParams["ytick.labelsize"] = 14
    matplotlib.rcParams["legend.fontsize"] = 11
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
end


function build_color_map(models::Vector{String})
    colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    return Dict(model => colors[(i - 1) % length(colors)] for (i, model) in enumerate(models))
end


function validate_columns!(df::DataFrame, required_cols::Vector{String}, csv_path::AbstractString)
    missing_cols = [col for col in required_cols if !hasproperty(df, Symbol(col))]
    isempty(missing_cols) || throw(ArgumentError("Missing required columns in $csv_path: $(join(missing_cols, ", "))"))
end


function load_offdiags_data(csv_path::AbstractString)::DataFrame
    df = CSV.read(csv_path, DataFrame)
    validate_columns!(df, ["model", "L", "n_nz"], csv_path)
    df.L = Int.(round.(Float64.(df.L)))
    df.n_nz = Float64.(df.n_nz)
    sort!(df, [:model, :L])
    return df
end


function linear_fit(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    length(xs) >= 2 || throw(ArgumentError("At least two points are needed for a linear fit."))
    coeff = hcat(Float64.(xs), ones(length(xs))) \ Float64.(ys)
    return coeff[1], coeff[2]
end


function plot_offdiags_nnz_vs_L(;
    input_csv::AbstractString=INPUT_CSV,
    output_basename::AbstractString=OUTPUT_BASENAME,
    fit_Lmin::Real=14,
    figure_size::Tuple{<:Real, <:Real}=(6.8, 4.6),
    use_custom_style::Bool=true,
    show_plot::Bool=true,
)
    isfile(input_csv) || throw(ArgumentError("CSV file not found: $input_csv"))

    apply_offdiags_style!(; use_custom_style=use_custom_style)
    df = load_offdiags_data(input_csv)
    models = [model for model in MODEL_ORDER if model in Set(String.(df.model))]
    color_map = build_color_map(models)

    fig, ax = pyplot.subplots(figsize=figure_size)

    for model in models
        subdf = df[String.(df.model) .== model, :]
        xs = Float64.(subdf.L)
        ys = Float64.(subdf.n_nz)
        fit_mask = xs .>= Float64(fit_Lmin)
        slope, intercept = linear_fit(xs[fit_mask], ys[fit_mask])
        fit_label = @sprintf("%s: %.2f L %+.2f", MODEL_LABELS[model], slope, intercept)
        color = color_map[model]

        ax.scatter(
            xs,
            ys;
            color=color,
            marker=MODEL_MARKERS[model],
            s=28,
            label=fit_label,
            zorder=3,
        )

        fit_x = collect(range(minimum(xs), maximum(xs); length=200))
        ax.plot(
            fit_x,
            slope .* fit_x .+ intercept;
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )
    end

    ax.set_xlabel(raw"$L$")
    ax.set_ylabel(raw"$n_{\rm{nz}}$")
    ax.set_xticks(sort(unique(Int.(df.L))))
    ax.grid(false)
    ax.legend(frameon=false, loc="upper left")

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

    return nothing
end


plot_offdiags_nnz_vs_L()
