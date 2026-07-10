
function binary_scm(d, d_first, ρ = 0.01)

    dgp = @dgp(
       C ~ Uniform(-1, 1),
       L ~ SklarDist(GaussianCopula(d, ρ), Tuple(fill(Uniform(-1,1), d))),
       p = sqrt(d_first) .* (vec(mean(L[:,1:d_first], dims = 2))),
       A ~ Bernoulli.(logistic.(2 .* p)),
       μ = A .* abs.(C) .+ p,
       Y ~ Normal.(μ, 1.0)
    )

    scm = StructuralCausalModel(dgp, :A, :Y)

    # Numerically approximate the mean part involving L integrated out by the CATE
    #monte_carlo = rand(dgp, 10^6)
    #partial_mean = sqrt(d_first) .* mean(2 .* monte_carlo.L[:, 1:d_first] .- (monte_carlo.L[:, 1:d_first] .^2))

    cate(C) = abs.(C)
    
    return scm, cate
end


function safe_predict(mach, X,  miny, maxy)
    preds = MLJ.predict(mach, X)
    preds[preds .< miny] .= miny
    preds[preds .> maxy] .= maxy
    return preds
end

function simulate_binom(scm::StructuralCausalModel, cate, n::Int, modellist, dir, i, grid_size)
    result = []
    grid_result = []

    true_ate, true_eff_bound = ate(scm)

    # Generate training data
    ct = rand(scm, n)
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
            cate_mach = machine(outcome_model, C, eif) |> fit!
            cate_pred = MLJ.predict(cate_mach, Ctest)
            cate_mse = mean((cate_pred .- cate(Ctest.C)).^2)

            # Get CATE predictions on a grid
            C_grid = range(0, 1, grid_size)
            cate_pred_grid = MLJ.predict(cate_mach, (C = C_grid,))

            push!(grid_result, (
                C = C_grid, n = fill(n, grid_size), model_name = fill(model_pair[1], grid_size), preds = cate_pred_grid
            ))
            
            push!(result, (
                n = n, model_name = model_pair[1], plugin = plugin,
                mse_outcome = mse_outcome, mse_propensity = mse_propensity, 
                ose = ose, ose_var = ose_var,
                true_ate = true_ate, true_eff_bound = true_eff_bound,
                cate_mse = cate_mse,
                time_outcome = time_outcome, time_propensity = time_propensity
            ))
    end

    # Save results as CSV after each thread completes
    sv = "n=" * string(n) * "_s=" * string(i)
    sv_grid = sv * "_preds.csv" 
    sv *= "_metrics.csv"

    output_dir = datadir(dir)
    isdir(output_dir) || mkpath(output_dir)

    CSV.write(joinpath(output_dir, sv_grid), reduce(vcat, DataFrame(gr) for gr in grid_result))
    CSV.write(joinpath(output_dir, sv), DataFrame(result))
    
    return result
end