using DrWatson
@quickactivate :RandomHALsims
using Random
using RandomHAL
using StatsBase
using Tables
using CausalTables
using Makie
using CairoMakie
using MLJ
using LogExpFunctions
using Distributions, Copulas

d = 40
d_first = 4

scm, cate = binary_scm(d, d_first)

n = 1600
ct = rand(scm, n)
mean(ct.arrays.μ)
hist(ct.data.A)
hist(ct.arrays.μ)
hist(ct.arrays.p)
hist(ct.data.Y)
pr = logistic.(ct.arrays.p / sqrt(d_first))
maximum(pr)
minimum(pr)
hist(pr)

scatter(conmean(scm, ct, :Y), ct.data.Y, color = ct.data.A)
mean(ct.data.Y[ct.data.A .== 1]) - mean(ct.data.Y[ct.data.A .== 0])
ate(scm)


XA = responseparents(ct)
X = treatmentparents(ct)
A = treatmentmatrix(ct)[:,1]
y = responsematrix(ct)[:, 1]
C = (C = Tables.getcolumn(X, 1),)
miny = minimum(y)
maxy = maximum(y)


# Generate testing data
cttest = rand(scm, n)
XAtest = responseparents(cttest)
Xtest = treatmentparents(cttest)
Atest = treatmentmatrix(ct)[:,1]
ytest = responsematrix(cttest)[:, 1]
Ctest = (C = Tables.getcolumn(Xtest, 1),)

# Get true function values
true_conmean = conmean(scm, cttest, :Y)
true_prob = conmean(scm, cttest, :A)

ct_A1 = intervene(ct, treat_all)
XA_A1 = responseparents(ct_A1)
ct_A0 = intervene(ct, treat_none)
XA_A0 = responseparents(ct_A0)

kwargs = (standardize = false, nlambda = 100, nfolds = 10,)
m = Int(round(0.3 * n * log(n)))
outcome_model = RandomHALRegressor(0, m, NamedTuple(), kwargs)
propensity_model = RandomHALBinaryClassifier(0, m, NamedTuple(), kwargs)


# Fit models
time_outcome = @elapsed outcome_mach = machine(outcome_model, XA, y) |> fit!
time_propensity = @elapsed propensity_mach = machine(propensity_model, X, A) |> fit!


p = scatter(true_conmean, MLJ.predict(outcome_mach, XAtest))
p = scatter(true_prob, MLJ.predict(propensity_mach, Xtest))

# Estimate performance of models on new data
mse_outcome = mean((RandomHALsims.safe_predict(outcome_mach, XAtest, miny, maxy) .- true_conmean).^2)
mse_propensity = mean((MLJ.predict(propensity_mach, Xtest) .- true_prob).^2)

# Compute one-step estimates using models
prA = RandomHALsims.safe_predict(propensity_mach, X, 0.02, 0.98)
μ = RandomHALsims.safe_predict(outcome_mach, XA, miny, maxy)
μ1 = RandomHALsims.safe_predict(outcome_mach, XA_A1, miny, maxy)
μ0 = RandomHALsims.safe_predict(outcome_mach, XA_A0, miny, maxy)
eif = μ1 - μ0 + ((A ./ prA) .- ((1 .- A) ./ (1 .- prA))) .* (y .- μ)


plugin = mean(μ1 .- μ0)
ose = mean(eif)
ose_var = var(eif) / n

ose_var .* 1600

# Compute CATE
cate_mach = machine(outcome_model, C, eif) |> fit!
#cate_mach = machine(HALRegressor(), C, eif) |> fit!

cate_pred = MLJ.predict(cate_mach, Ctest)
cate_mse = mean((cate_pred .- cate(Ctest.C)).^2)

grid_size = 101
C_grid = range(0, 1, grid_size)
true_cate_grid = cate(C_grid)
cate_pred_grid = MLJ.predict(cate_mach, (C = C_grid,))

fig, ax, line_plot = lines(C_grid, true_cate_grid)
stairs!(ax, C_grid, cate_pred_grid)
scatter!(ax, C_grid, cate_pred_grid)
fig



