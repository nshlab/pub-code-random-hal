using DrWatson
@quickactivate "RandomHALsims"

using CSV
using DataFrames
using DataFramesMeta
using Tables
using Statistics
using CairoMakie
using AlgebraOfGraphics

create_axis_time(ns, title) = (aspect=1, xticks=ns, xlabel = "Sample size", ylabel="Training time (seconds)", title = title)
create_axis_ose(ns, ylabel) = (aspect=1, xticks=ns, xlabel = "Sample size", ylabel=ylabel)
create_axis_mse(ns, title) = (aspect=1, xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = title)

function generate_plots(df_raw, str)
    df_raw[!, "upper"] = df_raw.ose .+ 1.96 .* sqrt.(df_raw.ose_var)
    df_raw[!, "lower"] = df_raw.ose .- 1.96 .* sqrt.(df_raw.ose_var)

    df = @chain df_raw begin
        @groupby(:n, :model_name)
        @combine(:mean_mse_outcome = mean(:mse_outcome), 
                :mean_mse_propensity = mean(:mse_propensity), 
                :mean_bias = mean(:ose) .- mean(:true_ate), 
                :mean_ose_var = mean(:ose_var),
                :mean_cate_mse = mean(:cate_mse),
                :mean_time_outcome = mean(:time_outcome),
                :mean_time_propensity = mean(:time_propensity),
                :coverage = mean((:true_ate .< :upper) .&& (:true_ate .> :lower))
                )
    end

    # Add some extra variables
    df[!, "smoothness"] = SubString.(df.model_name, length.(df.model_name))
    df[!, "model"]  = SubString.(df.model_name, 1, length.(df.model_name) .- 1)
    df[!, "scaled_mse_outcome"] = df.mean_mse_outcome .* sqrt.(df.n)

    df[!, "mean_bias"] = abs.(df.mean_bias)
    df[!, "scaled_bias"] = df.mean_bias .* sqrt.(df.n)
    df[!, "scaled_mse"] = df.n .* ((df.mean_bias .^ 2) .+ df.mean_ose_var)

    template = data(df) * visual(Lines)

    # Figure 1
    fig = Figure(; size=(600, 400))
    p1 = template * 
        mapping(:n, :mean_mse_outcome, color=:model, linestyle=:smoothness)

    p2 = template * 
        mapping(:n, :mean_mse_propensity, color=:model, linestyle=:smoothness)

    ag = draw!(fig[1, 1], p1, axis=create_axis_mse(ns, "Outcome model"))
    ag = draw!(fig[1, 2], p2, axis=create_axis_mse(ns, "Propensity model"))
    legend!(fig[2, 1:2], ag, orientation=:horizontal, tellheight=true)
    save(plotsdir(str*"MSE.png"), fig)

    # Figure 2
    fig = Figure(; size=(600, 600))

    p1 = template * 
        mapping(:n, :mean_bias, color=:model, linestyle=:smoothness)
    ag = draw!(fig[1, 1], p1, axis=create_axis_ose(ns, "Bias"))

    p2 = template * 
        mapping(:n, :scaled_bias, color=:model, linestyle=:smoothness)
    ag = draw!(fig[1, 2], p2, axis=create_axis_ose(ns, "Scaled bias"))

    p3 = template * 
        mapping(:n, :scaled_mse, color=:model, linestyle=:smoothness)
    ag = draw!(fig[2, 1], p3, axis=create_axis_ose(ns, "Scaled MSE"))

    p4 = (template * mapping(:n, :coverage, color=:model, linestyle=:smoothness)) + 
        (visual(HLines) * mapping([0.95]))
    ag = draw!(fig[2, 2], p4, axis=(aspect=1, xticks=ns, yticks = [0.95, 0.70, 0.45, 0.2, 0.0], xlabel = "Sample size", ylabel="Coverage"))

    legend!(fig[3, 1:2], ag, orientation=:horizontal, tellheight=true)
    colgap!(fig.layout, 0)
    save(plotsdir(str*"onestep.png"), fig)

    # Figure 3
    p = template * 
        mapping(:n, :mean_cate_mse, color=:model, linestyle=:smoothness)
    fig = draw(p, axis=(aspect=1, xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = "CATE Estimate"))
    save(plotsdir(str*"cate.png"), fig)

    # Figure 4
    fig = Figure(; size=(600, 400))
    p1 = template * 
        mapping(:n, :mean_time_outcome, color=:model, linestyle=:smoothness)

    p2 = template * 
        mapping(:n, :mean_time_propensity, color=:model, linestyle=:smoothness)


    ag = draw!(fig[1, 1], p1, axis=create_axis_time(ns, "Outcome model"))
    ag = draw!(fig[1, 2], p2, axis=create_axis_time(ns, "Propensity model"))
    legend!(fig[2, 1:2], ag, orientation=:horizontal, tellheight=true)
    save(plotsdir(str*"time.png"), fig)
end


# Comparison 
name(n) = "iters=200_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=$(n).csv"

ns = [100, 400, 900, 1600]
result = [CSV.read(datadir(name(n)), DataFrame) for n in ns]
df_raw = DataFrame(reduce(vcat, result))

generate_plots(df_raw, "compare_")

### Large variables test ###
name(n) = "iters=5_models=RandomHAL0_RandomHAL1_n=$(n).csv"

ns = [100]
result = [CSV.read(datadir(name(n)), DataFrame) for n in ns]
df_raw = DataFrame(reduce(vcat, result))

generate_plots(df_raw, "test_")


