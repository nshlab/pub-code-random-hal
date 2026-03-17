
function binary_scm(d, d_first, ρ = 0.05, treat_shift = 2)

    dgp = @dgp(
        L ~ SklarDist(GaussianCopula(d, ρ), Tuple(fill(Beta(2,2), d))),
        μ = (1 .+ 2 .* L[:, 1]) .* vec(mean(L[:,2:d_first] .- L[:,2:d_first] .^ (1/2), dims = 2)) .+ 0.5,
        A ~ Bernoulli.(logistic.(μ)),
        Y ~ Normal.((1 .+ treat_shift .* A) .* μ .+ 2, 0.1)
    )

    scm = StructuralCausalModel(dgp, :A, :Y)
    cate(L) = treat_shift .* ((1 .+ 2 .* L[:, 1]) .* mean(vec(mean(L[:,2:d_first] .- (L[:,2:d_first] .^(1/2)), dims = 2))) .+ 0.5)
    
    return scm, cate
end

function safe_predict(mach, X,  miny, maxy)
    preds = MLJ.predict(mach, X)
    preds[preds .< miny] .= miny
    preds[preds .> maxy] .= maxy
    return preds
end

function simulate_binom(scm::StructuralCausalModel, cate, n::Int, iters::Int, modellist)
    result = []

    true_ate, true_eff_bound = ate(scm)

    Threads.@threads for i in 1:iters
        println("Iteration $i, Thread $(Threads.threadid())")
        # Generate training data
        ct = rand(scm, n)
        XA = responseparents(ct)
        X = treatmentparents(ct)
        A = treatmentmatrix(ct)[:,1]
        y = responsematrix(ct)[:, 1]
        X1 = (X1 = Tables.getcolumn(X, 1),)
        miny = minimum(y)
        maxy = maximum(y)


        # Generate testing data
        cttest = rand(scm, n)
        XAtest = responseparents(cttest)
        Xtest = treatmentparents(cttest)
        Atest = treatmentmatrix(ct)[:,1]
        ytest = responsematrix(cttest)[:, 1]
        X1test = (X1 = Tables.getcolumn(Xtest, 1),)

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
                mse_outcome = mean((safe_predict(outcome_mach, XAtest, miny, maxy) .- true_conmean).^2)
                mse_propensity = mean((MLJ.predict(propensity_mach, Xtest) .- true_prob).^2)

                # Compute one-step estimates using models
                prA = safe_predict(propensity_mach, X, 0.02, 0.98)
                μ = safe_predict(outcome_mach, XA, miny, maxy)
                μ1 = safe_predict(outcome_mach, XA_A1, miny, maxy)
                μ0 = safe_predict(outcome_mach, XA_A0, miny, maxy)
                eif = μ1 - μ0 + ((A ./ prA) .- ((1 .- A) ./ (1 .- prA))) .* (y .- μ)


                plugin = mean(μ1 .- μ0)
                ose = mean(eif)
                ose_var = var(eif) / n

                # Compute CATE
                cate_mach = machine(outcome_model, X1, eif) |> fit!
                cate_pred = MLJ.predict(cate_mach, X1test)
                cate_mse = mean((cate_pred .- cate(Tables.matrix(Xtest))).^2)

                push!(result, (
                    n = n, model_name = model_pair[1], plugin = plugin,
                    mse_outcome = mse_outcome, mse_propensity = mse_propensity, 
                    ose = ose, ose_var = ose_var,
                    true_ate = true_ate, true_eff_bound = true_eff_bound,
                    cate_mse = cate_mse,
                    time_outcome = time_outcome, time_propensity = time_propensity
                ))
        end
    end

        # Save results as CSV after each thread completes
    sv = DrWatson.savename((; n, iters, models=join([m[1] for m in modellist], "_")), "csv")
    CSV.write(datadir(string(Dates.now()) * "_" * sv), DataFrame(result))

    return result
end