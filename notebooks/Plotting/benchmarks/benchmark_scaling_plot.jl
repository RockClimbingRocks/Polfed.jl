using DrWatson
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

using CSV
using DataFrames
using PythonPlot

const BENCHMARK_MODEL_ORDER = [
    "xxz",
    "syk4-d3",
    "j1-j2",
    "j1-j2-j3",
    "syk4-d4",
    "syk4-d5",
    "syk4-d6",
]

const BENCHMARK_MARKERS = Dict(
    "xxz" => "o",
    "syk4-d3" => "s",
    "j1-j2" => "^",
    "j1-j2-j3" => "D",
    "syk4-d4" => "P",
    "syk4-d5" => "X",
    "syk4-d6" => "v",
)


function loglog_power_fit(xvals::AbstractVector{<:Real}, yvals::AbstractVector{<:Real})
    length(xvals) == length(yvals) || throw(ArgumentError("xvals and yvals must have the same length."))
    length(xvals) >= 2 || throw(ArgumentError("Need at least two points for a log-log fit."))

    logx = log.(Float64.(xvals))
    logy = log.(Float64.(yvals))

    xmean = sum(logx) / length(logx)
    ymean = sum(logy) / length(logy)
    slope = sum((logx .- xmean) .* (logy .- ymean)) / sum((logx .- xmean) .^ 2)
    intercept = ymean - slope * xmean

    return intercept, slope
end


function benchmark_dataset_path(dataset)
    dataset_key = dataset isa Symbol ? dataset : lowercase(String(dataset))

    if dataset_key in (:polfed, :benchmark, :base, "polfed", "benchmark", "base", "benchmark_polfed.csv")
        return joinpath(@__DIR__, "benchmark_polfed.csv")
    elseif dataset isa AbstractString
        return isabspath(dataset) ? dataset : joinpath(@__DIR__, dataset)
    end

    throw(ArgumentError("Unknown dataset selector '$dataset'. Use :polfed, :benchmark, or pass a CSV filename/path."))
end


function apply_benchmark_style!(; use_custom_style::Bool=true)
    if use_custom_style
        try
            PythonPlot.pyplot.style.use(["rok-custom"])
        catch
        end
    end
    PythonPlot.matplotlib.rcParams["figure.dpi"] = 1000
end


function ordered_models(models::AbstractVector{<:AbstractString}; model_order::Vector{String}=BENCHMARK_MODEL_ORDER)
    present = Set(models)
    return [model for model in model_order if model in present]
end


function validate_models!(df::DataFrame; model_order::Vector{String}=BENCHMARK_MODEL_ORDER)
    allowed = Set(model_order)
    unknown_models = sort(unique(String(model) for model in df.model if !(String(model) in allowed)))
    isempty(unknown_models) || throw(ArgumentError("Unsupported model names: $(join(unknown_models, ", "))"))
end


function build_color_map(L_values)
    colors = PythonPlot.matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    ordered_L = sort(unique(Int.(collect(L_values))))
    return Dict(L => colors[mod1(i, length(colors))] for (i, L) in enumerate(ordered_L))
end


function load_polfed_benchmark(dataset)::DataFrame
    csv_path = benchmark_dataset_path(dataset)
    isfile(csv_path) || throw(ArgumentError("CSV file not found: $csv_path"))

    df = CSV.read(csv_path, DataFrame)
    required_cols = ["L", "model", "x", "y"]
    missing_cols = [col for col in required_cols if !hasproperty(df, Symbol(col))]
    isempty(missing_cols) || throw(ArgumentError("Missing required columns in $csv_path: $(join(missing_cols, ", "))"))

    out = DataFrame(
        L=Int.(df.L),
        model=String.(df.model),
        x=Float64.(df.x),
        y=Float64.(df.y),
    )
    validate_models!(out)
    return out
end


function plot_benchmark_scaling_axis!(
    ax,
    df::DataFrame;
    color_map::Dict,
    marker_size::Real,
    line_width::Real,
    marker_map::Dict=BENCHMARK_MARKERS,
    model_order::Vector{String}=BENCHMARK_MODEL_ORDER,
    title::Union{Nothing, AbstractString}=nothing,
    ylabel::Union{Nothing, AbstractString}=nothing,
)
    model_rank = Dict(model => idx for (idx, model) in enumerate(model_order))
    local_df = copy(df)
    local_df.model_rank = [model_rank[String(model)] for model in local_df.model]
    sort!(local_df, [:L, :model_rank])

    for L in sort(unique(local_df.L))
        subdf = local_df[local_df.L .== L, :]
        color = color_map[L]
        xs = Float64.(subdf.x)
        ys = Float64.(subdf.y)
        models = String.(subdf.model)

        intercept, slope = loglog_power_fit(xs, ys)
        xfit = 10 .^ range(log10(minimum(xs)), log10(maximum(xs)); length=200)
        yfit = exp(intercept) .* xfit .^ slope
        ax.plot(
            xfit,
            yfit;
            color=color,
            linewidth=line_width,
            linestyle="--",
            alpha=0.9,
            zorder=1,
        )

        for idx in eachindex(xs)
            ax.scatter(
                [xs[idx]],
                [ys[idx]];
                s=marker_size,
                marker=marker_map[models[idx]],
                color=color,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=4,
            )
        end
    end

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(raw"$N_{\mathrm{OFF}}\ \mathrm{per\ row}$")
    ylabel !== nothing && ax.set_ylabel(ylabel)
    title !== nothing && ax.set_title(title)
    ax.grid(true; which="major", alpha=0.16, linewidth=0.5)
    ax.grid(true; which="minor", alpha=0.06, linewidth=0.35)
    ax.tick_params(which="both", direction="in")
    return ax
end


function add_benchmark_legends!(
    ax;
    color_map::Dict,
    model_order::Vector{String},
    marker_map::Dict=BENCHMARK_MARKERS,
    line_width::Real,
    legend_fontsize::Real=6,
    model_legend_anchor::Tuple{<:Real, <:Real}=(1.02, 0.70),
)
    line2d = PythonPlot.matplotlib.lines.Line2D

    ordered_L = sort(collect(keys(color_map)))
    size_handles = Any[
        line2d(
            [0],
            [0];
            color=color_map[L],
            linestyle="--",
            linewidth=line_width,
            label="L = $(L)",
        ) for L in ordered_L
    ]
    legend_sizes = ax.legend(
        size_handles,
        ["L = $(L)" for L in ordered_L];
        title="Sizes",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=false,
        borderaxespad=0.0,
        labelspacing=0.45,
        handlelength=2.0,
        handletextpad=0.6,
        fontsize=legend_fontsize,
    )
    ax.add_artist(legend_sizes)

    model_handles = Any[
        line2d(
            [0],
            [0];
            linestyle="None",
            marker=marker_map[model],
            markerfacecolor="#4a4a4a",
            markeredgecolor="#4a4a4a",
            markersize=4,
            label=model,
        ) for model in model_order
    ]
    ax.legend(
        model_handles,
        model_order;
        title="Model",
        loc="upper left",
        bbox_to_anchor=model_legend_anchor,
        frameon=false,
        borderaxespad=0.0,
        labelspacing=0.45,
        handletextpad=0.5,
        fontsize=legend_fontsize,
    )
end


function plot_benchmark_scaling(;
    dataset::Union{Symbol, AbstractString}=:polfed,
    savepath::Union{Nothing, AbstractString}=nothing,
    show_plot::Bool=true,
    marker_size::Real=30,
    line_width::Real=1.2,
    figure_size::Tuple{<:Real, <:Real}=(5.0, 2.0),
    use_custom_style::Bool=true,
)
    apply_benchmark_style!(; use_custom_style=use_custom_style)

    df = load_polfed_benchmark(dataset)
    color_map = build_color_map(df.L)
    fig, ax = PythonPlot.pyplot.subplots(figsize=figure_size)

    plot_benchmark_scaling_axis!(
        ax,
        df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        ylabel=raw"$t_{\mathrm{cpu}}$",
    )

    add_benchmark_legends!(
        ax;
        color_map=color_map,
        model_order=ordered_models(String.(df.model)),
        line_width=line_width,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))

    if savepath !== nothing
        fig.savefig(savepath; bbox_inches="tight")
    end
    if show_plot
        display(fig)
        fig.show()
    end
    return fig, ax, df
end


plot_benchmark_scaling()
