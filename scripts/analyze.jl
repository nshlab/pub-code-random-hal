using DrWatson
@quickactivate "RandomHALsims"

using CSV
using DataFrames
using DataFramesMeta
using Tables
using Statistics
using CairoMakie
using AlgebraOfGraphics

create_axis_time(ns) = (aspect = 1, xticks=ns, xlabel = "Sample size", ylabel="Training time (seconds)")
create_axis_ose(ns, ylabel) = (aspect=1, xticks=ns, xlabel = "Sample size", ylabel=ylabel)
create_axis_mse(ns, title) = (xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = title)

function generate_plots(df_raw, str, eff_bound, models, names)
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
    df[!, :model] = [names[findfirst(==(m), models)] for m in df.model]
    

    template = data(df) * visual(Lines, markersize=8) * visual(ScatterLines, markersize=8) * mapping(layout = :smoothness)
    set_theme!(palette = (color = [:black, "#0f5575", "#ffa600", "#4e7647", "#00aaf5", "#002e5c"],))
    
    # Figure 1
    fig = Figure(; size=(960, 640))
    p1 = template * 
        mapping(:n, :mean_mse_outcome, color=:model, linestyle=:smoothness, marker=:smoothness)

    p2 = template * 
        mapping(:n, :mean_mse_propensity, color=:model, linestyle=:smoothness, marker=:smoothness)

    ag = draw!(fig[1, 1:3], p1, axis=(aspect = 1, xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE"))
    ag = draw!(fig[3, 1:3], p2, axis=(aspect = 1, xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE"))
    fig[0, 2] = Label(fig, "Outcome regression", fontsize = 18, font = :bold)
    fig[2, 2] = Label(fig, "Propensity score", fontsize = 18, font = :bold)
    # reduce vertical gap between title row and plots
    legend!(fig[1, 4], ag, orientation=:vertical, tellheight=true)
    rowsize!(fig.layout, 0, 20)
    rowsize!(fig.layout, 1, 300)
    rowsize!(fig.layout, 2, 20)
    rowsize!(fig.layout, 3, 300)
    resize_to_layout!(fig)
    save(plotsdir(str*"MSE.png"), fig)

    # Figure 2
    fig = Figure(; size=(1500, 600))

    p1 = template * 
        mapping(:n, :mean_bias, color=:model, linestyle=:smoothness, marker=:smoothness)
    ag = draw!(fig[1, 1], p1, axis=create_axis_ose(ns, "Bias"))

    p2 = template * 
        mapping(:n, :scaled_bias, color=:model, linestyle=:smoothness, marker=:smoothness)
    ag = draw!(fig[1, 2], p2, axis=create_axis_ose(ns, "Scaled bias"))

    p3 = template * 
        mapping(:n, :scaled_mse, color=:model, linestyle=:smoothness, marker=:smoothness)# + 
        #(visual(HLines) * mapping([eff_bound]))
    ag = draw!(fig[2, 1], p3, axis=create_axis_ose(ns, "Scaled MSE / Eff. Bound"))

    p4 = (template * mapping(:n, :coverage, color=:model, linestyle=:smoothness, marker=:smoothness)) + 
        (visual(HLines) * mapping([0.95]))
    ag = draw!(fig[2, 2], p4, axis=(aspect=1, xticks=ns, yticks = [0.0, 0.5, 0.95], xlabel = "Sample size", ylabel="Coverage", limits=(nothing, (0.0, 1.0))))

    legend!(fig[1, 3], ag, orientation=:vertical, tellheight=true)
    
    rowsize!(fig.layout, 1, 300)
    rowsize!(fig.layout, 2, 300)
    resize_to_layout!(fig)
    #colgap!(fig.layout, 0)
    save(plotsdir(str*"onestep.png"), fig)

    # Figure 3
    fig = Figure(; size=(1000, 310))
    p = template * 
        mapping(:n, :mean_cate_mse, color=:model, linestyle=:smoothness, marker=:smoothness)
    draw!(fig[1,1:3], p, axis=(xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE"))
    fig[0, 2] = Label(fig, "CATE Estimate", fontsize = 18, font = :bold)
    rowsize!(fig.layout, 0, 10)
    rowsize!(fig.layout, 1, 300)
    resize_to_layout!(fig)
    legend!(fig[1, 4], ag, orientation=:vertical, tellheight=true)

    save(plotsdir(str*"cate.png"), fig)

    # Figure 4
    fig = Figure(; size=(960, 640))
    p1 = template * 
        mapping(:n, :mean_time_outcome, color=:model, linestyle=:smoothness, marker=:smoothness)

    p2 = template * 
        mapping(:n, :mean_time_propensity, color=:model, linestyle=:smoothness, marker=:smoothness)

    ag = draw!(fig[1, 1:3], p1, axis=create_axis_time(ns))
    ag = draw!(fig[3, 1:3], p2, axis=create_axis_time(ns))
    legend!(fig[1, 4], ag, orientation=:vertical, tellheight=true)
    fig[0, 2] = Label(fig, "Outcome regression", fontsize = 18, font = :bold)
    fig[2, 2] = Label(fig, "Propensity score", fontsize = 18, font = :bold)
    rowsize!(fig.layout, 0, 20)
    rowsize!(fig.layout, 1, 300)
    rowsize!(fig.layout, 2, 20)
    rowsize!(fig.layout, 3, 300)
    resize_to_layout!(fig)

    legend!(fig[1, 4], ag, orientation=:vertical, tellheight=true)
    save(plotsdir(str*"time.png"), fig)
end

filenames = [
    "3_small_comparison-combined.csv"
]
models = ["HAL", "RandomHAL", "RandomHAL_intdecay", "RandomHAL_keeptreat"]
names = ["HAL", "RandomHAL — uniform sampling", "RandomHAL — low-order interactions more likely", "RandomHAL — always sample treatment"]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = sort(DataFrame(reduce(vcat, result)), :n)

generate_plots(df_raw, "small0520_", mean(df_raw.true_eff_bound), models, names)

### Large variables test ###
filenames = [
    "4_large_randomhal-combined.csv"
]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = sort(DataFrame(reduce(vcat, result)), :n)

models = ["RandomHAL", "RandomHAL_intdecay", "RandomHAL_keeptreat"]
names = ["RandomHAL — uniform sampling", "RandomHAL — low-order interactions more likely", "RandomHAL — always sample treatment"]

generate_plots(df_raw, "large0520_", mean(df_raw.true_eff_bound), models, names)


df = @chain df_raw begin
    @groupby(:n, :model_name)
    @combine(:mean_mse_outcome = mean(:mse_outcome), 
            :mean_mse_propensity = mean(:mse_propensity), 
            :mean_bias = mean(:ose) .- mean(:true_ate), 
            :mean_ose_var = mean(:ose_var),
            :mc_var = var(:ose),
            :mean_cate_mse = mean(:cate_mse),
            :mean_time_outcome = mean(:time_outcome),
            :mean_time_propensity = mean(:time_propensity)#,
            #:coverage = mean((:true_ate .< :upper) .&& (:true_ate .> :lower))
            )
end


