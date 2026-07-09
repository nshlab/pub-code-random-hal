using DrWatson
@quickactivate "RandomHALsims"

using RandomHALsims
using CSV
using DataFrames
using DataFramesMeta
using CategoricalArrays
using Tables
using Statistics
using CairoMakie
using AlgebraOfGraphics

create_axis_time(ns) = (aspect = 1, xticks=ns, xlabel = "Sample size", ylabel="Training time (seconds)")
create_axis_ose(ns, ylabel) = (aspect=1, xticks=ns, xlabel = "Sample size", ylabel=ylabel)
create_axis_mse(ns, title) = (xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = title)
function configure_layout_axes!(layout, colgap, rowgap)
    for spec in layout.content
        child = spec.content
        
        if child isa Axis
            child.xlabel = "Sample size"        
            child.xlabelvisible = true          
            child.xticklabelsvisible = true      
            
        elseif child isa GridLayout
            # Tighten the space between facets
            colgap!(child, colgap)
            rowgap!(child, rowgap)

            # Continue scanning deeper layouts if nested
            configure_layout_axes!(child, colgap, rowgap)
        end
    end
end
configure_layout_axes!(layout) = configure_layout_axes!(layout, 10, 10)

function generate_plots(df_raw, str, eff_bound, models, names, max_n = 1600)
    ns = unique(df_raw.n)
    df_raw[!, "upper"] = df_raw.ose .+ 1.96 .* sqrt.(df_raw.ose_var)
    df_raw[!, "lower"] = df_raw.ose .- 1.96 .* sqrt.(df_raw.ose_var)

    df = @chain df_raw begin
        @groupby(:n, :model_name)
        @combine(:mean_mse_outcome = mean(:mse_outcome), 
                :mean_mse_propensity = mean(:mse_propensity), 
                :mean_bias = mean(:ose) .- mean(:true_ate), 
                :mean_ose_var = mean(:ose_var),
                :mc_var = var(:ose),
                :mean_cate_mse = mean(:cate_mse),
                :mean_time_outcome = mean(:time_outcome),
                :mean_time_propensity = mean(:time_propensity),
                :coverage = mean((:true_ate .< :upper) .&& (:true_ate .> :lower))
                )
    end

    # Add some extra variables
    df[!, "smoothness"] = "Smoothness = " .* SubString.(df.model_name, length.(df.model_name))
    df[!, "model"]  = SubString.(df.model_name, 1, length.(df.model_name) .- 1)
    df[!, "scaled_mse_outcome"] = df.mean_mse_outcome .* sqrt.(df.n)

    df[!, "mean_bias"] = abs.(df.mean_bias)
    df[!, "scaled_bias"] = df.mean_bias .* sqrt.(df.n)
    df[!, "scaled_mse"] = df.n .* ((df.mean_bias .^ 2) .+ df.mc_var) ./ eff_bound

    df = filter(row -> row.model in models, df)
    df = filter(row -> row.n <= max_n, df)

    df[!, :model] = [names[findfirst(==(m), models)] for m in df.model]
    df[!, :model] = CategoricalArrays.categorical(df.model; ordered=true, levels=names)
    
    # Build the scale config to hide BOTH LineStyle and Marker from the legend engine
    # This is because we already facet by smoothness
    
    hidden_scales = scales(
        LineStyle = (; legend = false),
        Marker = (; legend = false)
    )
    template = data(df) * visual(Lines, linewidth=2.5) * visual(ScatterLines, markersize=10, linewidth=2.5) * mapping(layout = :smoothness)
    set_theme!(
        fontsize = 20,#16
        linewidth = 5,#2.5
        markersize = 20,#10
        Axis = (
            xlabelsize = 20,
            ylabelsize = 20,
            xticklabelsize = 18,
            yticklabelsize = 18
        ),
        Legend = (
            labelsize = 20,
            titlesize = 20
        ),
        palette = (color = ["#0f5575", "#ffa600", "#4e7647", :black, "#00aaf5", "#002e5c"],)
    )

    # Figure 1
    fig = Figure(; size=(640, 640))
    p1 = template * 
        mapping(:n => "", :mean_mse_outcome, color=:model => "", linestyle=:smoothness, marker=:smoothness)

    p2 = template * 
        mapping(:n => "", :mean_mse_propensity, color=:model => "", linestyle=:smoothness, marker=:smoothness)

    # Draw commands utilizing the scales object
    ag = draw!(fig[1, 1:3], p1, hidden_scales; axis=(aspect = 1, xticks=ns, ylabel="Out-of-sample MSE"), facet = (; linkxaxes = :none))
    ag = draw!(fig[3, 1:3], p2, hidden_scales; axis=(aspect = 1, xticks=ns, ylabel="Out-of-sample MSE"), facet = (; linkxaxes = :none))

    fig[0, 2] = Label(fig, "Outcome regression", fontsize = 20, font = :bold)
    fig[2, 2] = Label(fig, "Propensity score", fontsize = 20, font = :bold)
    legend!(fig[4, 1:3], ag, orientation=:vertical, tellheight=true)
    configure_layout_axes!(fig.layout, 20, 10)

    # reduce vertical gap between title row and plots
    rowsize!(fig.layout, 0, 10)
    rowsize!(fig.layout, 1, 300)
    rowsize!(fig.layout, 2, 10)
    rowsize!(fig.layout, 3, 300)

    resize_to_layout!(fig)
    save(plotsdir(str*"MSE.png"), fig)

    # Figure 2
    fig = Figure(; size=(1500, 600))

    p1 = template * 
        mapping(:n => "", :mean_bias, color=:model => "", linestyle=:smoothness, marker=:smoothness)
    ag = draw!(fig[1, 1], p1, hidden_scales, axis=(aspect=1, xticks=ns, ylabel="Bias"), facet = (; linkxaxes = :none))

    p2 = template * 
        mapping(:n => "", :scaled_bias, color=:model => "", linestyle=:smoothness, marker=:smoothness)
    ag = draw!(fig[1, 2], p2, hidden_scales, axis=(aspect=1, xticks=ns, ylabel="Scaled bias"), facet = (; linkxaxes = :none))

    p3 = template * 
        mapping(:n => "", :scaled_mse, color=:model => "", linestyle=:smoothness, marker=:smoothness)# + 
        #(visual(HLines) * mapping([eff_bound]))
    ag = draw!(fig[2, 1], p3, hidden_scales, axis=(aspect=1, xticks=ns, ylabel="Scaled MSE / Eff. Bound"), facet = (; linkxaxes = :none))

    p4 = (visual(HLines) * mapping([0.95])) + (template * mapping(:n => "", :coverage, color=:model => "", linestyle=:smoothness, marker=:smoothness))
      # Single positional argument as required by HLines

    ag = draw!(fig[2, 2], p4, hidden_scales, 
          axis=(aspect=1, xticks=ns, yticks = [0.0, 0.5, 0.95], ylabel="Coverage", limits=(nothing, (0.0, 1.0))), 
          facet = (; linkxaxes = :none))
    legend!(fig[3, 1:2], ag, orientation=:vertical, tellheight=true)
    
    configure_layout_axes!(fig.layout)
    rowsize!(fig.layout, 1, 300)
    rowsize!(fig.layout, 2, 300)
    resize_to_layout!(fig)
    save(plotsdir(str*"onestep.png"), fig)

    # Figure 3
    fig = Figure(; size=(1000, 310))
    p = template * 
        mapping(:n => "", :mean_cate_mse, color=:model => "", linestyle=:smoothness, marker=:smoothness)
    draw!(fig[1,1:3], p, hidden_scales, axis=(xticks=ns, ylabel="Out-of-sample MSE"))
    fig[0, 2] = Label(fig, "CATE Estimate", fontsize = 20, font = :bold)
    legend!(fig[2, 1:3], ag, orientation=:vertical, tellheight=true)
    configure_layout_axes!(fig.layout, 20, 10)

    rowsize!(fig.layout, 0, 10)
    rowsize!(fig.layout, 1, 300)
    resize_to_layout!(fig)
    save(plotsdir(str*"cate.png"), fig)

    # Figure 4
    fig = Figure(; size=(640, 640))
    p1 = template * 
        mapping(:n => "", :mean_time_outcome, color=:model => "", linestyle=:smoothness, marker=:smoothness)

    p2 = template * 
        mapping(:n => "", :mean_time_propensity, color=:model => "", linestyle=:smoothness, marker=:smoothness)

    ag = draw!(fig[1, 1:3], p1, hidden_scales, axis=(aspect = 1, xticks=ns, ylabel="Training time (seconds)"))
    ag = draw!(fig[3, 1:3], p2, hidden_scales, axis=(aspect = 1, xticks=ns, ylabel="Training time (seconds)"))
    legend!(fig[4, 1:3], ag, orientation=:vertical, tellheight=true)
    fig[0, 2] = Label(fig, "Outcome regression", fontsize = 20, font = :bold)
    fig[2, 2] = Label(fig, "Propensity score", fontsize = 20, font = :bold)
    configure_layout_axes!(fig.layout, 20, 10)
    rowsize!(fig.layout, 0, 10)
    rowsize!(fig.layout, 1, 300)
    rowsize!(fig.layout, 2, 10)
    rowsize!(fig.layout, 3, 300)
    resize_to_layout!(fig)
    save(plotsdir(str*"time.png"), fig)
end

function generate_pred_plots(df_raw, str, models, names, d, d_first, n)

    # Get the true CATE function for the DGP
    scm, cate = binary_scm(d, d_first)

    # Compute summary statistics across the grid and filter the data
    df = @chain df_raw begin
            @groupby(:n, :model_name, :C)
            @combine(
                :mean_pred = mean(:preds),
                :var_pred = var(:preds)
            )
        end
    df[!, "smoothness"] = "Smoothness = " .* SubString.(df.model_name, length.(df.model_name))
    df[!, "model"]  = SubString.(df.model_name, 1, length.(df.model_name) .- 1)
    df = filter(row -> row.model in models, df)
    df[!, "upper"] = df.mean_pred .+ (1.96 .* sqrt.(df.var_pred))
    df[!, "lower"] = df.mean_pred .- (1.96 .* sqrt.(df.var_pred))

    df[!, :model] = [names[findfirst(==(m), models)] for m in df.model]
    df[!, :model] = CategoricalArrays.categorical(df.model; ordered=true, levels=vcat(["True CATE"], names))

    df0 = filter(row -> (row.model == "RandomHAL — uniform sampling") & (row.smoothness == "Smoothness = 0") & (row.n == n), df)
    df1 = filter(row -> (row.model == "RandomHAL — uniform sampling") & (row.smoothness == "Smoothness = 1") & (row.n == n), df)

    # Create staircase effect for df0 by duplicating each row
    df0_sorted = sort(df0, :C)
    df0_stairs = []
    for i in 1:nrow(df0_sorted)
        row = df0_sorted[i, :]
        push!(df0_stairs, row)
        if i < nrow(df0_sorted)
            next_row = (n = row.n, model_name = row.model_name, var_pred = row.var_pred, smoothness = row.smoothness, model = row.model, C = df0_sorted[i+1, :C], mean_pred = row.mean_pred, upper = row.upper, lower = row.lower)
            push!(df0_stairs, next_row)
        end
    end
    df0 = DataFrame(df0_stairs)

    # Set up plotting
    set_theme!(
        fontsize = 20,#16
        linewidth = 5,#2.5
        markersize = 20,#10
        Axis = (
            xlabelsize = 20,
            ylabelsize = 20,
            xticklabelsize = 18,
            yticklabelsize = 18
        ),
        Legend = (
            labelsize = 20,
            titlesize = 20
        ),
        palette = (color = [:black, "#0f5575", "#ffa600"],)
    )

    fig = Figure(; size=(1200, 500))
    hidden_scales = scales(
        LineStyle = (; legend = false),
    )

    # Create AlgebraOfGraphics templates and plots for the different smoothness levels
    mean_template0 = data(df0) * visual(Lines, linewidth=2.5)
    mean_template1 = data(df1) * visual(Lines, linewidth=2.5)
    cb_template0 = data(df0) * visual(Band, alpha = 0.3)
    cb_template1 = data(df1) * visual(Band, alpha = 0.3)

    p0 = (mean_template0 * mapping(:C, :mean_pred, color=:model, linestyle=:smoothness)) +
        (cb_template0 * mapping(:C, :upper, :lower, color=:model))
    p1 = (mean_template1 * mapping(:C, :mean_pred, color=:model, linestyle=:smoothness)) +
        (cb_template1 * mapping(:C, :upper, :lower, color=:model))

    # Plot the truth
    C_grid = unique(df.C)
    true_df = DataFrame((C = C_grid, true_cate = cate.(C_grid)))
    true_df[!, "model"] = CategoricalArrays.categorical(fill("True CATE", nrow(true_df)); ordered=true, levels=vcat(["True CATE"], names))
    true_template = data(true_df) * visual(Lines, linewidth=2.5)
    p_true = true_template * mapping(:C, :true_cate, color=:model)

    # Combine the plots together
    ag = draw!(fig[1,1], p_true + p0, hidden_scales, axis=(xlabel = "Covariate value", ylabel="CATE"))#, limits = ((0.0, 1.0), (-2.1, 1.1))))
    ag = draw!(fig[1,2], p_true + p1, hidden_scales, axis=(xlabel = "Covariate value", ylabel=""))#, limits = ((0.0, 1.0), (-2.1, 1.1))))

    fig[0, 1] = Label(fig, "Smoothness = 0", fontsize = 20, font = :bold)
    fig[0, 2] = Label(fig, "Smoothness = 1", fontsize = 20, font = :bold)

    legend!(fig[2, 1:2], ag, orientation=:vertical, tellheight=true)

    colsize!(fig.layout, 1, 400)
    colsize!(fig.layout, 2, 400)
    rowsize!(fig.layout, 0, 10)
    rowsize!(fig.layout, 1, 240)
    rowsize!(fig.layout, 2, 100)

    resize_to_layout!(fig)
    save(plotsdir(str * "cate_preds.png"), fig)
end

### Small Comparison ###
filenames = [
    "3_small_comparison-combined-metrics (13).csv"
]
models = ["RandomHAL", "RandomHAL_intdecay", "RandomHAL_keeptreat", "HAL"]
names = ["RandomHAL — uniform sampling", "RandomHAL — low-order interactions more likely", "RandomHAL — always sample treatment", "HAL"]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = sort(DataFrame(reduce(vcat, result)), :n)

generate_plots(df_raw, "small_", mean(df_raw.true_eff_bound), models, names)

### Small Comparison CATE ###
filenames = [
    "3_small_comparison-combined-preds (13).csv"
]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = sort(DataFrame(reduce(vcat, result)), :n)

generate_pred_plots(df_raw, "small_", models, names, 4, 4, 1600)


### Large Comparison ###
filenames = [
    "4_large_randomhal-combined-metrics (13).csv"
]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = sort(DataFrame(reduce(vcat, result)), :n)

models = ["RandomHAL", "RandomHAL_intdecay", "RandomHAL_keeptreat"]
names = ["RandomHAL — uniform sampling", "RandomHAL — low-order interactions more likely", "RandomHAL — always sample treatment"]

generate_plots(df_raw, "large_", mean(df_raw.true_eff_bound), models, names)


### Large Comparison CATE ###
filenames = [
    "4_large_randomhal-combined-preds (13).csv"
]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = sort(DataFrame(reduce(vcat, result)), :n)

generate_pred_plots(df_raw, "large_", models, names, 40, 8, 1600)







