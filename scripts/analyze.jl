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

function generate_plots(df_raw, str, eff_bound)
    ns = unique(df_raw.n)
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
        mapping(:n, :scaled_mse, color=:model, linestyle=:smoothness) + 
        (visual(HLines) * mapping([eff_bound]))
    ag = draw!(fig[2, 1], p3, axis=create_axis_ose(ns, "Scaled MSE"))

    p4 = (template * mapping(:n, :coverage, color=:model, linestyle=:smoothness)) + 
        (visual(HLines) * mapping([0.95]))
    ag = draw!(fig[2, 2], p4, axis=(aspect=1, xticks=ns, yticks = [0.0, 0.5, 0.95], xlabel = "Sample size", ylabel="Coverage", limits=(nothing, (0.0, 1.0))))

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
#name(n) = "iters=200_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=$(n).csv"
#ns = [100, 400, 900, 1600]
#result = [CSV.read(datadir(name(n)), DataFrame) for n in ns]

filenames = [
    "2026-03-17T15_16_28.950_iters=100_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=100.csv",
    "2026-03-17T16_57_04.174_iters=100_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=400.csv",
    "2026-03-17T21_07_20.152_iters=100_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=900.csv",
    "2026-03-18T08_55_40.667_iters=100_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=1600.csv"
]

filenames = [
    "2026-05-13T14:11:50.026_iters=2_models=RandomHAL0_RandomHAL1_RandomHAL_minnonzero0_RandomHAL_minnonzero1_RandomHAL_keeptreat0_RandomHAL_keeptreat1_RandomHAL_intdecay0_RandomHAL_intdecay1_HAL0_HAL1_n=100.csv",
    "2026-05-13T14:18:36.004_iters=2_models=RandomHAL0_RandomHAL1_RandomHAL_minnonzero0_RandomHAL_minnonzero1_RandomHAL_keeptreat0_RandomHAL_keeptreat1_RandomHAL_intdecay0_RandomHAL_intdecay1_HAL0_HAL1_n=400.csv"
]

result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = DataFrame(reduce(vcat, result))

generate_plots(df_raw, "test_compare_", 0.0964)

### Large variables test ###
filenames = [
    "2026-03-20T15_55_06.191_iters=100_models=RandomHAL0_RandomHAL1_n=100.csv",
    "2026-03-20T18_11_19.043_iters=100_models=RandomHAL0_RandomHAL1_n=400.csv",
    "2026-03-21T01_30_26.486_iters=100_models=RandomHAL0_RandomHAL1_n=900.csv",
    "2026-03-21T21_56_35.340_iters=100_models=RandomHAL0_RandomHAL1_n=1600.csv"
]

filenames = [
    "2026-05-13T16:24:03.578_iters=2_models=RandomHAL0_RandomHAL1_RandomHAL_minnonzero0_RandomHAL_minnonzero1_RandomHAL_keeptreat0_RandomHAL_keeptreat1_RandomHAL_intdecay0_RandomHAL_intdecay1_n=100.csv",
    "2026-05-13T16:27:12.006_iters=2_models=RandomHAL0_RandomHAL1_RandomHAL_minnonzero0_RandomHAL_minnonzero1_RandomHAL_keeptreat0_RandomHAL_keeptreat1_RandomHAL_intdecay0_RandomHAL_intdecay1_n=400.csv"
]


result = [CSV.read(datadir(name), DataFrame) for name in filenames]
df_raw = DataFrame(reduce(vcat, result))

generate_plots(df_raw, "test_compare_", 0.075)


