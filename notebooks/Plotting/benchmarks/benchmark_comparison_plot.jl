const COMPARISON_HELPERS = (
    :benchmark_dataset_path,
    Symbol("apply_benchmark_style!"),
    :build_color_map,
    :load_polfed_benchmark,
    :loglog_power_fit,
    Symbol("add_benchmark_legends!"),
    Symbol("validate_models!"),
)

if !all(name -> isdefined(@__MODULE__, name), COMPARISON_HELPERS)
    include(joinpath(@__DIR__, "benchmark_scaling_plot.jl"))
end

const COMPARISON_MODEL_ORDER = [
    "xxz",
    "syk4-d2",
    "syk4-d3",
    "j1-j2",
    "j1-j2-j3",
    "syk4-d4",
    "syk4-d5",
]

const COMPARISON_MARKERS = merge(
    BENCHMARK_MARKERS,
    Dict(
        "syk4-d2" => "*",
    ),
)


function shiftinvert_dataset_path(dataset)
    dataset_key = dataset isa Symbol ? dataset : lowercase(String(dataset))

    if dataset_key in (:shiftinvert, :shift_and_invert, "shiftinvert", "shift_and_invert", "benchmark_shift-and-invert.csv")
        return joinpath(@__DIR__, "benchmark_shift-and-invert.csv")
    elseif dataset isa AbstractString
        return isabspath(dataset) ? dataset : joinpath(@__DIR__, dataset)
    end

    throw(ArgumentError("Unknown shift-and-invert dataset selector '$dataset'."))
end


function canonical_shiftinvert_model(model, d)
    model_name = String(model)
    model_name == "syk4" || return model_name
    ismissing(d) && throw(ArgumentError("Missing `d` for syk4 row in shift-and-invert benchmark data."))
    return "syk4-d$(Int(round(Float64(d))))"
end


function load_shiftinvert_benchmark(dataset::Union{Symbol, AbstractString}="benchmark_shift-and-invert.csv")::DataFrame
    csv_path = shiftinvert_dataset_path(dataset)
    isfile(csv_path) || throw(ArgumentError("CSV file not found: $csv_path"))

    df = CSV.read(csv_path, DataFrame)
    required_cols = ["L", "model", "x", "wall_time", "mpi_cpus"]
    missing_cols = [col for col in required_cols if !hasproperty(df, Symbol(col))]
    isempty(missing_cols) || throw(ArgumentError("Missing required columns in $csv_path: $(join(missing_cols, ", "))"))

    dcol = hasproperty(df, :d) ? df.d : fill(missing, nrow(df))
    out = DataFrame(
        L=Int.(df.L),
        model=[canonical_shiftinvert_model(model, d) for (model, d) in zip(df.model, dcol)],
        x=Float64.(df.x),
        y=Float64.(df.wall_time) .* Float64.(df.mpi_cpus),
    )
    out = filter_comparison_models(out)
    validate_models!(out; model_order=COMPARISON_MODEL_ORDER)
    return out
end


function ordered_comparison_models(models::AbstractVector{<:AbstractString})
    present = Set(models)
    return [model for model in COMPARISON_MODEL_ORDER if model in present]
end


function model_legend_label(model::AbstractString)
    return uppercase(model)
end


function filter_comparison_models(df::DataFrame)::DataFrame
    allowed = Set(COMPARISON_MODEL_ORDER)
    return df[[String(model) in allowed for model in df.model], :]
end


function filter_by_Lmin(df::DataFrame, Lmin)::DataFrame
    Lmin === nothing && return df
    return df[df.L .>= Lmin, :]
end


function overlay_benchmark_method!(
    ax,
    df::DataFrame;
    color_map::Dict,
    marker_size::Real,
    line_width::Real,
    alpha::Real,
    Lmin::Union{Nothing, Integer}=18,
)
    model_rank = Dict(model => idx for (idx, model) in enumerate(COMPARISON_MODEL_ORDER))
    local_df = filter_by_Lmin(copy(df), Lmin)
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
            alpha=alpha,
            zorder=1,
        )

        for idx in eachindex(xs)
            ax.scatter(
                [xs[idx]],
                [ys[idx]];
                s=marker_size,
                marker=COMPARISON_MARKERS[models[idx]],
                color=color,
                edgecolors="white",
                linewidths=0.5,
                alpha=alpha,
                zorder=4,
            )
        end
    end
end


function plot_benchmark_comparison(;
    polfed_dataset::Union{Symbol, AbstractString}="benchmark_polfed.csv",
    shift_dataset::Union{Symbol, AbstractString}="benchmark_shift-and-invert.csv",
    Lmin::Union{Nothing, Integer}=18,
    savepath::Union{Nothing, AbstractString}="/home/rokpintar/projects/Polfed/plots/benchmarks/polfed_vs_shift-and-invert.pdf",
    show_plot::Bool=true,
    marker_size::Real=30,
    line_width::Real=1.2,
    figure_size::Tuple{<:Real, <:Real}=(6.0, 2.5),
    use_custom_style::Bool=true,
)
    apply_benchmark_style!(; use_custom_style=use_custom_style)

    polfed_df = filter_comparison_models(load_polfed_benchmark(polfed_dataset))
    shift_df = load_shiftinvert_benchmark(shift_dataset)
    effective_Lmin = isnothing(Lmin) ? 10 : Lmin
    plot_polfed_df = filter_by_Lmin(polfed_df, effective_Lmin)
    plot_shift_df = filter_by_Lmin(shift_df, effective_Lmin)
    color_map = build_color_map(vcat(plot_polfed_df.L, plot_shift_df.L))
    fig, ax = PythonPlot.pyplot.subplots(figsize=figure_size)

    overlay_benchmark_method!(
        ax,
        polfed_df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        alpha=0.95,
        Lmin=effective_Lmin,
    )
    overlay_benchmark_method!(
        ax,
        shift_df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        alpha=0.5,
        Lmin=effective_Lmin,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(raw"$N_{\mathrm{OFF}}\ \mathrm{per\ row}$")
    ax.set_ylabel(raw"$t_{\mathrm{cpu}}$")
    ax.set_title("CPU time - POLFED VS shift-and-invert")
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(11)
    ax.xaxis.labelpad = 1.5
    ax.grid(false; which="major", alpha=0.16, linewidth=0.5)
    ax.grid(false; which="minor", alpha=0.06, linewidth=0.35)
    ax.tick_params(which="both", direction="in")

    add_benchmark_legends!(
        ax;
        color_map=color_map,
        model_order=ordered_comparison_models(vcat(String.(plot_polfed_df.model), String.(plot_shift_df.model))),
        marker_map=COMPARISON_MARKERS,
        line_width=line_width,
    )

    line2d = PythonPlot.matplotlib.lines.Line2D
    model_order = ordered_comparison_models(vcat(String.(plot_polfed_df.model), String.(plot_shift_df.model)))
    model_handles = Any[
        line2d(
            [0],
            [0];
            linestyle="None",
            marker=COMPARISON_MARKERS[model],
            markerfacecolor="#4a4a4a",
            markeredgecolor="#4a4a4a",
            markersize=4,
            label=model,
        ) for model in model_order
    ]
    ax.legend(
        model_handles,
        [model_legend_label(model) for model in model_order];
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=false,
        borderaxespad=0.0,
        labelspacing=0.45,
        handletextpad=0.5,
        fontsize=6,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))

    if savepath !== nothing
        println("dwlijfwohdqhwldw")
        fig.savefig(savepath; bbox_inches="tight")
    end
    if show_plot
        display(fig)
        fig.show()
    end
    # return fig, ax, plot_polfed_df, plot_shift_df
end


plot_benchmark_comparison()
