using DrWatson
@quickactivate "RandomHALsims"

using CSV
using DataFrames
using DataFramesMeta
using Tables
using Statistics

name(n) = "iters=5_models=RandomHAL0_RandomHAL1_HAL0_HAL1_n=$(n).csv"

result = [datadir(name(50)) for n in [50, 100, 400]]
CSV.read(datadir(name(50)), DataFrame)
result = [CSV.read(datadir(name(n)), DataFrame) for n in [50, 100, 400]]
df = DataFrame(reduce(vcat, result))

@chain df begin
    @groupby(:n, :model_name)
    @combine(:mean_mse_outcome = mean(:mse_outcome), 
             :mean_mse_propensity = mean(:mse_propensity), 
             :mean_bias = mean(:ose) .- mean(:true_ate), 
             :mean_ose_var = mean(:ose_var),
             :mean_cate_mse = mean(:cate_mse)
             )
end