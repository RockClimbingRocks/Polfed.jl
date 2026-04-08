using DrWatson
@quickactivate "notebooks"
ENV["JULIA_DEPOT_PATH"] = "~/.julia_interactive"

using CSV
using DataFrames
using PythonPlot

if !isdefined(@__MODULE__, :BENCHMARK_2P_MODEL_ORDER)
    const BENCHMARK_2P_MODEL_ORDER = [
        "xxz",
        "syk4-d2",
        "syk4-d3",
        "j1-j2",
        "j1-j2-j3",
        "syk4-d4",
        "syk4-d5",
    ]
end

if !isdefined(@__MODULE__, :BENCHMARK_2P_MARKERS)
    const BENCHMARK_2P_MARKERS = Dict(
        "xxz" => "o",
        "syk4-d2" => "*",
        "syk4-d3" => "s",
        "j1-j2" => "^",
        "j1-j2-j3" => "D",
        "syk4-d4" => "P",
        "syk4-d5" => "X",
    )
end


function apply_two_panel_style!(; use_custom_style::Bool=true)
    if use_custom_style
        try
            PythonPlot.pyplot.style.use(["rok-custom"])
        catch
        end
    end
    PythonPlot.matplotlib.rcParams["figure.dpi"] = 1000
end


function build_color_map(L_values)
    colors = PythonPlot.matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    ordered_L = sort(unique(Int.(collect(L_values))))
    return Dict(L => colors[mod1(i, length(colors))] for (i, L) in enumerate(ordered_L))
end


function validate_columns!(df::DataFrame, required_cols::Vector{String}, csv_path::AbstractString)
    missing_cols = [col for col in required_cols if !hasproperty(df, Symbol(col))]
    isempty(missing_cols) || throw(ArgumentError("Missing required columns in $csv_path: $(join(missing_cols, ", "))"))
end


function validate_models!(df::DataFrame)
    allowed = Set(BENCHMARK_2P_MODEL_ORDER)
    unknown_models = sort(unique(String(model) for model in df.model if !(String(model) in allowed)))
    isempty(unknown_models) || throw(ArgumentError("Unsupported model names: $(join(unknown_models, ", "))"))
end


function polfed_dataset_path(dataset)
    dataset_key = dataset isa Symbol ? dataset : lowercase(String(dataset))

    if dataset_key in (:polfed, "polfed", "benchmark_polfed.csv")
        return joinpath(@__DIR__, "benchmark_polfed.csv")
    elseif dataset isa AbstractString
        return isabspath(dataset) ? dataset : joinpath(@__DIR__, dataset)
    end

    throw(ArgumentError("Unknown Polfed dataset selector '$dataset'."))
end


function shiftinvert_dataset_path(dataset)
    dataset_key = dataset isa Symbol ? dataset : lowercase(String(dataset))

    if dataset_key in (:shiftinvert, :shift_and_invert, "shiftinvert", "shift_and_invert", "benchmark_shift-and-invert.csv")
        return joinpath(@__DIR__, "benchmark_shift-and-invert.csv")
    elseif dataset isa AbstractString
        return isabspath(dataset) ? dataset : joinpath(@__DIR__, dataset)
    end

    throw(ArgumentError("Unknown shift-and-invert dataset selector '$dataset'."))
end


function canonical_shiftinvert_model(model, model_full, d)
    if !ismissing(model_full) && !isempty(strip(String(model_full)))
        return String(model_full)
    end

    model_name = String(model)
    model_name == "syk4" || return model_name
    ismissing(d) && throw(ArgumentError("Missing `d` for syk4 row in shift-and-invert benchmark data."))
    return "syk4-d$(Int(round(Float64(d))))"
end


function filter_models(df::DataFrame)::DataFrame
    allowed = Set(BENCHMARK_2P_MODEL_ORDER)
    return df[[String(model) in allowed for model in df.model], :]
end


function filter_by_Lmin(df::DataFrame, Lmin)::DataFrame
    Lmin === nothing && return df
    return df[df.L .>= Lmin, :]
end


function ordered_models(models::AbstractVector{<:AbstractString})
    present = Set(models)
    return [model for model in BENCHMARK_2P_MODEL_ORDER if model in present]
end


function model_legend_label(model::AbstractString)
    return uppercase(model)
end


function load_polfed_benchmark(dataset::Union{Symbol, AbstractString}="benchmark_polfed.csv"; nev_divisor::Union{Nothing, Real}=nothing)::DataFrame
    csv_path = polfed_dataset_path(dataset)
    isfile(csv_path) || throw(ArgumentError("CSV file not found: $csv_path"))

    df = CSV.read(csv_path, DataFrame)
    validate_columns!(df, ["L", "model", "x", "y"], csv_path)

    yvals = Float64.(df.y)
    nev_divisor !== nothing && (yvals ./= Float64(nev_divisor))

    out = DataFrame(
        L=Int.(df.L),
        model=String.(df.model),
        x=Float64.(df.x),
        y=yvals,
    )
    out = filter_models(out)
    validate_models!(out)
    return out
end


function load_shiftinvert_benchmark(dataset::Union{Symbol, AbstractString}="benchmark_shift-and-invert.csv"; nev_divisor::Union{Nothing, Real}=nothing)::DataFrame
    csv_path = shiftinvert_dataset_path(dataset)
    isfile(csv_path) || throw(ArgumentError("CSV file not found: $csv_path"))

    df = CSV.read(csv_path, DataFrame)
    validate_columns!(df, ["L", "model", "x", "wall_time", "mpi_cpus"], csv_path)

    model_full_col = hasproperty(df, :model_full) ? df.model_full : fill(missing, nrow(df))
    dcol = hasproperty(df, :d) ? df.d : fill(missing, nrow(df))
    yvals = Float64.(df.wall_time) .* Float64.(df.mpi_cpus)
    nev_divisor !== nothing && (yvals ./= Float64(nev_divisor))

    out = DataFrame(
        L=Int.(df.L),
        model=[canonical_shiftinvert_model(model, model_full, d) for (model, model_full, d) in zip(df.model, model_full_col, dcol)],
        x=Float64.(df.x),
        y=yvals,
    )
    out = filter_models(out)
    validate_models!(out)
    return out
end


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


function overlay_method!(
    ax,
    df::DataFrame;
    color_map::Dict,
    marker_size::Real,
    line_width::Real,
    alpha::Real,
    Lmin::Union{Nothing, Integer}=18,
    line_style="-",
    filled_markers::Bool=true,
)
    local_df = filter_by_Lmin(copy(df), Lmin)
    model_rank = Dict(model => idx for (idx, model) in enumerate(BENCHMARK_2P_MODEL_ORDER))
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
            linestyle=line_style,
            alpha=alpha,
            zorder=1,
        )

        for idx in eachindex(xs)
            if filled_markers
                ax.scatter(
                    [xs[idx]],
                    [ys[idx]];
                    s=marker_size,
                    marker=BENCHMARK_2P_MARKERS[models[idx]],
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    alpha=alpha,
                    zorder=4,
                )
            else
                ax.scatter(
                    [xs[idx]],
                    [ys[idx]];
                    s=marker_size,
                    marker=BENCHMARK_2P_MARKERS[models[idx]],
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.0,
                    alpha=alpha,
                    zorder=4,
                )
            end
        end
    end
end


function add_legends!(
    ax;
    color_map::Dict,
    model_order::Vector{String},
    line_width::Real,
)
    line2d = PythonPlot.matplotlib.lines.Line2D

    ordered_L = sort(collect(keys(color_map)))
    size_handles = Any[
        line2d(
            [0],
            [0];
            color=color_map[L],
            linestyle="-",
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
        fontsize=6,
    )
    ax.add_artist(legend_sizes)

    model_handles = Any[
        line2d(
            [0],
            [0];
            linestyle="None",
            marker=BENCHMARK_2P_MARKERS[model],
            markerfacecolor="#4a4a4a",
            markeredgecolor="#4a4a4a",
            markersize=4,
            label=model,
        ) for model in model_order
    ]
    legend_models = ax.legend(
        model_handles,
        [model_legend_label(model) for model in model_order];
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.6),
        frameon=false,
        borderaxespad=0.0,
        labelspacing=0.45,
        handletextpad=0.5,
        fontsize=6,
    )
    ax.add_artist(legend_models)

    method_handles = Any[
        line2d(
            [0],
            [0];
            color="#4a4a4a",
            linestyle="-",
            linewidth=line_width,
            alpha=1.0,
            marker="o",
            markerfacecolor="#4a4a4a",
            markeredgecolor="#4a4a4a",
            markersize=4,
            label="POLFED",
        ),
        line2d(
            [0],
            [0];
            color="#4a4a4a",
            linestyle="--",
            linewidth=line_width,
            alpha=1.0,
            marker="o",
            markerfacecolor="none",
            markeredgecolor="#4a4a4a",
            markersize=4,
            label="Shift-and-invert",
        ),
    ]
    legend_method = ax.legend(
        method_handles,
        ["POLFED", "Shift-And-Invert"];
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.075),
        frameon=false,
        borderaxespad=0.0,
        labelspacing=0.45,
        handlelength=2.0,
        handletextpad=0.6,
        fontsize=6,
    )
    legend_method.get_title().set_ha("left")
    legend_method.get_title().set_position((-150.0, 0.0))
end


function format_panel!(ax, ylabel::AbstractString; title::Union{Nothing, AbstractString}=nothing)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(raw"$N_{\mathrm{OFF}}\ \mathrm{per\ row}$")
    ax.set_ylabel(ylabel)
    if title !== nothing
        ax.set_title(title)
    end
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(11)
    ax.xaxis.labelpad = 1.5
    ax.grid(false)
    ax.tick_params(which="both", direction="in")
end


function plot_benchmark_comparison_per_ev(;
    polfed_dataset::Union{Symbol, AbstractString}="benchmark_polfed.csv",
    shift_dataset::Union{Symbol, AbstractString}="benchmark_shift-and-invert.csv",
    polfed_nev::Real=1500,
    shift_nev::Real=100,
    Lmin::Union{Nothing, Integer}=18,
    savepath::Union{Nothing, AbstractString}="/home/rokpintar/projects/Polfed/plots/benchmarks/polfed_vs_shift-and-invert_two_panel.pdf",
    show_plot::Bool=true,
    marker_size::Real=30,
    line_width::Real=1.2,
    figure_size::Tuple{<:Real, <:Real}=(7.2, 2.8),
    use_custom_style::Bool=true,
)
    apply_two_panel_style!(; use_custom_style=use_custom_style)

    polfed_total_df = load_polfed_benchmark(polfed_dataset)
    shift_total_df = load_shiftinvert_benchmark(shift_dataset)
    polfed_per_ev_df = load_polfed_benchmark(polfed_dataset; nev_divisor=polfed_nev)
    shift_per_ev_df = load_shiftinvert_benchmark(shift_dataset; nev_divisor=shift_nev)

    effective_Lmin = isnothing(Lmin) ? 18 : Lmin
    color_map = build_color_map(vcat(
        filter_by_Lmin(polfed_total_df, effective_Lmin).L,
        filter_by_Lmin(shift_total_df, effective_Lmin).L,
        filter_by_Lmin(polfed_per_ev_df, effective_Lmin).L,
        filter_by_Lmin(shift_per_ev_df, effective_Lmin).L,
    ))

    fig, axs = PythonPlot.pyplot.subplots(1, 2; figsize=figure_size)
    ax_left = axs[0]
    ax_right = axs[1]

    overlay_method!(
        ax_left,
        polfed_total_df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        alpha=1.0,
        Lmin=effective_Lmin,
        line_style="-",
        filled_markers=true,
    )
    overlay_method!(
        ax_left,
        shift_total_df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        alpha=1.0,
        Lmin=effective_Lmin,
        line_style="--",
        filled_markers=false,
    )
    format_panel!(
        ax_left,
        "";
        title="CPU time",
    )

    overlay_method!(
        ax_right,
        polfed_per_ev_df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        alpha=1.0,
        Lmin=effective_Lmin,
        line_style="-",
        filled_markers=true,
    )
    overlay_method!(
        ax_right,
        shift_per_ev_df;
        color_map=color_map,
        marker_size=marker_size,
        line_width=line_width,
        alpha=1.0,
        Lmin=effective_Lmin,
        line_style="--",
        filled_markers=false,
    )
    format_panel!(
        ax_right,
        "";
        title="CPU time per eigenvalue",
    )

    add_legends!(
        ax_right;
        color_map=color_map,
        model_order=ordered_models(vcat(
            String.(filter_by_Lmin(polfed_total_df, effective_Lmin).model),
            String.(filter_by_Lmin(shift_total_df, effective_Lmin).model),
            String.(filter_by_Lmin(polfed_per_ev_df, effective_Lmin).model),
            String.(filter_by_Lmin(shift_per_ev_df, effective_Lmin).model),
        )),
        line_width=line_width,
    )

    # fig.tight_layout(rect=(0.0, 0.0, 0.8, 0.93))
    # fig.tight_layout(rect=(0.0, 0.0, 0., 0.))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.28)

    if savepath !== nothing
        fig.savefig(savepath; bbox_inches="tight")
    end
    if show_plot
        display(fig)
        fig.show()
    end
    return fig, axs, polfed_total_df, shift_total_df, polfed_per_ev_df, shift_per_ev_df
end


plot_benchmark_comparison_per_ev()
