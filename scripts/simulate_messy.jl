using DrWatson
@quickactivate "RandomHALsims"

using RandomHAL
using CausalTables
using Distributions
using LogExpFunctions
using MLJ
using Plots
using MLJXGBoostInterface
using MLJLinearModels
using Tables
using Copulas
using LinearAlgebra
using CSV
using DataFrames
using DataFramesMeta
using Tables


function binary_scm(d, d_first, ρ, treat_shift = 0.5)

    dgp = @dgp(
        L ~ SklarDist(GaussianCopula(d, ρ), Tuple(fill(Uniform(), d))),
        wave = wave(L, d_first),
        g = (1 .+ 2 .* L[:, 1].^2) .* wave,
        A ~ Bernoulli.(logistic.(2 .* (g .- 1.5))),
        Y ~ Normal.((1 .+ treat_shift .* A) .* g, 0.1)
    )

    scm = StructuralCausalModel(dgp, :A, :Y)
    cate(L) = treat_shift .* (1 .+ 2 .* L[:, 1].^2) .* mean(vec(mean(sin.(2 .* pi .* L[:, 2:d_first]), dims = 2)) .+ 1)
    
    return scm, cate
end

n = 900
d = 3
d_first = 3
scm, cate = binary_scm(d, d_first, 0.1)
ct = rand(scm, n)
XA = responseparents(ct)
X = treatmentparents(ct)
A = treatmentmatrix(ct)[:,1]
y = responsematrix(ct)[:, 1]
true_conmean = conmean(scm, ct, :Y)
true_prob = conmean(scm, ct, :A)

true_ate, eff_bound = ate(scm)

cttest = rand(scm, n)
XAtest = responseparents(cttest)
Xtest = responseparents(cttest)
Atest = treatmentmatrix(cttest)[:,1]
ytest = responsematrix(cttest)[:, 1]
true_conmean_test = conmean(scm, cttest, :Y)
true_prob_test = conmean(scm, cttest, :A)

outcome_model = RandomHALRegressor(smoothness = 1, max_block_size = n ÷ 2, tol = 1e-7, nfolds = 10, nlambda = 100)
t = @elapsed outcome_mach =  machine(outcome_model, XA, y) |> fit!
pred_conmean = predict(outcome_mach, XA)
mean((true_conmean .- pred_conmean).^2)
pred_conmean_test = predict(outcome_mach, XAtest)
mean((true_conmean_test .- pred_conmean_test).^2)

scatter(true_conmean, pred_conmean)
scatter(true_conmean_test, pred_conmean_test)

propensity_model = RandomHALClassifier(smoothness = 1, max_block_size = n ÷ 2, tol = 1e-7, nfolds = 5, nlambda = 50, min_λ_ε = 0.001)
t = @elapsed propensity_mach =  machine(propensity_model, X, Float64.(A)) |> fit!
pred_prob = predict(propensity_mach, X)
pred_prob_test = predict(propensity_mach, Xtest)
mean((true_prob_test .- pred_prob_test).^2)

scatter(true_prob, pred_prob)
scatter(true_prob_test, pred_prob_test)

ct_A1 = intervene(ct, treat_all)
XA_A1 = responseparents(ct_A1)
ct_A0 = intervene(ct, treat_none)
XA_A0 = responseparents(ct_A0)

prA = MLJ.predict(propensity_mach, X)
μ = MLJ.predict(outcome_mach, XA)
μ1 = MLJ.predict(outcome_mach, XA_A1)
μ0 = MLJ.predict(outcome_mach, XA_A0)
eif = μ1 - μ0 + ((A ./ prA) .- ((1 .- A) ./ (1 .- prA))) .* (y .- μ)

L1 = (L1 = ct.data.L_1,)
L1test = (L1 = cttest.data.L_1,)
scatter(L[:, 1], cate)

cate_mach = machine(outcome_model, L1, eif) |> fit!
cate_pred = MLJ.predict(cate_mach, L1test)

L = Tables.matrix(treatmentparents(cttest))

scatter(L1test.L1, cate_pred)
scatter!(L1test.L1, cate(L))

mean((cate_pred .- cate(L)).^2)
# was 0.006...


par = propensity_mach.fitresult.params
B = BasisMatrixBlocks(par.indblocks, Tables.matrix(X))
pred_path = 1 ./ (1 .+ exp.(-((B * par.β_path) .+ par.β0_path)))

scatter(true_prob, pred_prob, xlabel = "True propensity", ylabel = "Predicted propensity", title = "Random HAL Propensity Score Estimation")
anim = @animate for i in 2:length(par.λ)
    scatter(true_prob, pred_path[:, i], label = "λ = $(round(par.λ[i], digits = 6))", legend=:bottomright)
end
gif(anim, "anim_fps15.gif", fps = 15)


Btest = BasisMatrixBlocks(par.indblocks, Tables.matrix(Xtest))
pred_path_test = 1 ./ (1 .+ exp.(-((Btest * par.β_path) .+ par.β0_path)))

scatter(true_prob_test, pred_prob_test, xlabel = "True propensity", ylabel = "Predicted propensity", title = "Random HAL Propensity Score Estimation")
anim = @animate for i in 1:length(par.λ)
    scatter(true_prob_test, pred_path_test[:, i], label = par.λ[i], legend=:bottomright)
end
gif(anim, "anim_fps15.gif", fps = 15)

i = 30
scatter(true_prob_test, pred_path_test[:, i], label = par.λ[i], legend=:bottomright)

outcome_model = RandomHALRegressor(smoothness = 1, max_block_size = n ÷ 5, tol = 1e-7, nfolds = 5, nlambda = 50)
t = @elapsed outcome_mach =  machine(outcome_model, XA, y) |> fit!
pred_conmean = predict(outcome_mach, XA)
mean((true_conmean .- pred_conmean).^2)
scatter(true_conmean, pred_conmean)

pred_conmean_test = predict(outcome_mach, XAtest)
mean((true_conmean_test .- pred_conmean_test).^2)
scatter(true_conmean_test, pred_conmean_test)



function simulate_binom(scm::StructuralCausalModel, n::Int, iters::Int, modellist)
    result = []

    Threads.@threads for i in 1:iters
        println("Iteration $i, Thread $(Threads.threadid())")
        # Generate training data
        ct = rand(scm, n)
        XA = responseparents(ct)
        X = treatmentparents(ct)
        A = treatmentmatrix(ct)[:,1]
        y = responsematrix(ct)[:, 1]

        # Generate testing data
        cttest = rand(scm, n)
        XAtest = responseparents(cttest)
        Xtest = responseparents(cttest)
        Atest = treatmentmatrix(ct)[:,1]
        ytest = responsematrix(cttest)[:, 1]

        # Get true function values
        true_conmean = conmean(scm, cttest, :Y)
        true_prob = conmean(scm, cttest, :A)

        ct_A1 = intervene(ct, treat_all)
        XA_A1 = responseparents(ct_A1)
        ct_A0 = intervene(ct, treat_none)
        XA_A0 = responseparents(ct_A0)

        for model_pair in modellist
                outcome_model, propensity_model = model_pair[2]

                # Fit models
                time_outcome = @elapsed outcome_mach = machine(outcome_model, XA, y) |> fit!
                time_propensity = @elapsed propensity_mach = machine(propensity_model, X, A) |> fit!

                # Estimate performance of models on new data
                mse_outcome = mean((MLJ.predict(outcome_mach, XAtest) .- true_conmean).^2)
                mse_propensity = mean((MLJ.predict(propensity_mach, Xtest) .- true_prob).^2)

                # Compute one-step estimates using models
                prA = MLJ.predict(propensity_mach, X)
                μ = MLJ.predict(outcome_mach, XA)
                μ1 = MLJ.predict(outcome_mach, XA_A1)
                μ0 = MLJ.predict(outcome_mach, XA_A0)
                eif = μ1 - μ0 + ((A ./ prA) .- ((1 .- A) ./ (1 .- prA))) .* (y .- μ)

                plugin = mean(μ1 .- μ0)
                ose = mean(eif)
                ose_var = var(eif) / n

                push!(result, (
                    n = n, model_name = model_pair[1], plugin = plugin,
                    mse_outcome = mse_outcome, mse_propensity = mse_propensity, 
                    ose = ose, ose_var = ose_var,
                    time_outcome = time_outcome, time_propensity = time_propensity
                ))
        end
    end

        # Save results as CSV after each thread completes
    sv = savename((; n, iters, models=join([m[1] for m in modellist], "_")), "csv")
    CSV.write(datadir(sv), DataFrame(result))

    return result
end

make_models(n, k) = [
    "RandomHAL0" => (
    RandomHALRegressor(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "RandomHAL1" => (
    RandomHALRegressor(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    )
]

scm, cate = binary_scm(40, 20, 0.1)
result = [simulate_binom(scm, cate, n, 5, make_models(n, 4)) for n in [50, 100]]

df = DataFrame(reduce(vcat, result))

@chain df begin
    @groupby(:n, :model_name)
    @combine(:mean_mse_outcome = mean(:mse_outcome), 
             :mean_mse_propensity = mean(:mse_propensity), 
             :mean_ose = mean(:ose), 
             :mean_ose_var = mean(:ose_var))
end


make_comparison(n, k) = [
    "RandomHAL0" => (
    RandomHALRegressor(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 0, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "RandomHAL1" => (
    RandomHALRegressor(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100),
    RandomHALClassifier(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 100)
    ),
    "HAL0" => (
    HALRegressor(0),
    HALBinaryClassifier(0)
    ),
    "HAL1" => (
    HALRegressor(1),
    HALBinaryClassifier(1)
    )
]

scm, cate = binary_scm(3, 3, 0.1)
result = [simulate_binom(scm, cate, n, 5, make_comparison(n, 4)) for n in [50, 100]]
