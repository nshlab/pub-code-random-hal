using DrWatson
@quickactivate

using RandomHAL
using CausalTables
using Distributions
using LogExpFunctions
using MLJ
using Plots
using MLJXGBoostInterface
using MLJLinearModels

function binary_scm(d, d_first)
    covariates = DataGeneratingProcess(
        [O -> Beta(2,2) for _ in 1:d]
    )

    treatment_outcome = @dgp(
        A ~ Bernoulli.(logistic.(reduce(+, values(O)[1:d_first]) ./ 2)),
        Y ~ Normal.(reduce(+, values(O)[1:d_first]) ./ 2 .+ A, 0.2)
    )

    dgp = merge(covariates, treatment_outcome)
    scm = StructuralCausalModel(dgp, :A, :Y)
    return scm
end

n = 900
scm = binary_scm(40, 10)
ct = rand(scm, n)
XA = responseparents(ct)
X = treatmentparents(ct)
A = treatmentmatrix(ct)[:,1]
y = responsematrix(ct)[:, 1]

cttest = rand(scm, n)
XAtest = responseparents(cttest)
Xtest = responseparents(cttest)
Atest = treatmentmatrix(cttest)[:,1]
ytest = responsematrix(cttest)[:, 1]
true_conmean = conmean(scm, cttest, :Y)

# Let's compare the timing to a tuned XGBoost model
model = XGBoostRegressor()
r = range(model, :eta, lower=0.001, upper=1.0, scale=:log);
tuned_model = TunedModel(
    model=model,
    resampling=CV(nfolds=5),
    tuning=Grid(resolution=10),
    range=r,
    measure=rms
);
t, mach = machine(tuned_model, XA, y) |> fit!

model = RandomHALRegressor(smoothness = 0, max_block_size = 20, tol = 1e-7, nfolds = 5, nlambda = 50)
t = @elapsed mach =  machine(model, XA, y) |> fit!

mean((true_conmean .- predict(mach, XAtest)).^2)
scatter(predict(mach, XAtest), true_conmean, label = "Predicted vs True Conditional Mean", xlabel = "Predicted Conditional Mean", ylabel = "True Conditional Mean")

function simulate_binom(scm::StructuralCausalModel, n::Int, iters::Int, models)

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
            true_propensity = propensity(scm, cttest, :A)

            ct_A1 = intervene(ct, treat_all)
            XA_A1 = responseparents(ct_A1)
            ct_A0 = intervene(ct, treat_none)
            XA_A0 = responseparents(ct_A0)
    
            for model_pair in models
                    outcome_model, propensity_model = model_pair[2]

                    # Fit models
                    time_outcome = @elapsed outcome_mach = machine(outcome_model, XA, y) |> fit!
                    time_propensity = @elapsed propensity_mach = machine(propensity_model, X, Float64.(A)) |> fit!

                    # Estimate performance of models on new data
                    mse_outcome = mean((MLJ.predict(outcome_mach, XAtest) .- true_conmean).^2)
                    mse_propensity = mean((MLJ.predict(propensity_mach, Xtest) .- true_propensity).^2)
    
                    # Compute one-step estimates using models
                    prA = MLJ.predict(propensity_mach, X)
                    μ = MLJ.predict(outcome_mach, XA)
                    μ1 = MLJ.predict(outcome_mach, XA_A1)
                    μ0 = MLJ.predict(outcome_mach, XA_A0)
                    eif = μ1 - μ0 + ((A ./ prA) .- ((1 .- A) ./ (1 .- prA))) .* (y .- μ)

                    ose = mean(eif)
                    ose_var = var(eif) / n

                    push!(result, (n = n, model_name = model_pair[1], mse_outcome = mse_outcome, mse_propensity = mse_propensity, ose = ose, ose_var = ose_var))
            end
        end
        return result
end

make_models(n, k) = [
    "RandomHAL" => (
    RandomHALRegressor(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 50),
    RandomHALClassifier(smoothness = 1, max_block_size = n ÷ k, tol = 1e-7, nfolds = 5, nlambda = 50)
    )
]

result = [simulate_binom(scm, n, 5, make_models(n, 5)) for n in [100, 900, 2500]]

