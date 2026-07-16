
function binary_scm(d, d_first, ρ = 0.05)

    dgp = @dgp(
       C ~ Uniform(0, 1),
       L ~ SklarDist(GaussianCopula(d, ρ), Tuple(fill(Uniform(-1,1), d))),
       p = sqrt(d_first) .* (vec(mean(sin.(0.5 .* pi .* L[:,1:d_first]), dims = 2))),
       A ~ Bernoulli.(logistic.(2 .* p)),
       μ = A .* (sin.(0.7 .* pi .* C).^3) .+ p,
       Y ~ Normal.(μ, 1.0)
    )

    scm = StructuralCausalModel(dgp, :A, :Y)

    # Numerically approximate the mean part involving L integrated out by the CATE
    #monte_carlo = rand(dgp, 10^6)
    #partial_mean = sqrt(d_first) .* mean(2 .* monte_carlo.L[:, 1:d_first] .- (monte_carlo.L[:, 1:d_first] .^2))
    cate(C) = sin.(0.7 .* pi .* C).^3
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

    # Split ct into three folds (list of three CausalTables)
    inds = randperm(n)
    nfold = fld(n, 3)
    s = [nfold, nfold, n - 2 * nfold]
    ct = [rand(scm, n) for n in s]

    XA = responseparents.(ct)
    X = treatmentparents.(ct)
    A = getindex.(treatmentmatrix.(ct), :, 1)
    y = getindex.(responsematrix.(ct), :, 1)
    C = reduce(vcat, (C = Tables.getcolumn(Xs, 1),) for Xs in X])
    miny = minimum.(y)
    maxy = maximum.(y)


    # Generate testing data
    cttest = [rand(scm, n) for n in s]
    XAtest = responseparents.(cttest)
    Xtest = treatmentparents.(cttest)
    Atest = getindex.(treatmentmatrix.(ct), :,1)
    ytest = getindex.(responsematrix.(cttest), :, 1)
    Ctest = reduce(vcat, (C = Tables.getcolumn(Xtests, 1),) for Xtests in Xtest)

    # Get true function values
    true_conmean = map(cttest -> conmean(scm, cttest, :Y), cttest)
    true_prob = map(cttest -> conmean(scm, cttest, :A), cttest)

    ct_A1 = map(ct -> intervene(ct, treat_all), ct)
    XA_A1 = map(ct_A1 -> responseparents(ct_A1), ct_A1)
    ct_A0 = map(ct -> intervene(ct, treat_none), ct)
    XA_A0 = map(ct_A0 -> responseparents(ct_A0), ct_A0)


    for model_pair in modellist
            outcome_model, propensity_model = model_pair[2]

            # Fit models
            time_outcome = @elapsed outcome_mach = [machine(outcome_model, XA[i], y[i]) |> fit! for i in 1:3]
            time_propensity = @elapsed propensity_mach = [machine(propensity_model, X[i], A[i]) |> fit! for i in 1:3]

            # Estimate performance of models on new data
            oos_outcome_preds = [safe_predict(outcome_mach[i], XAtest[i], miny[i], maxy[i]) for i in 1:3]
            oos_propensity_preds = [MLJ.predict(propensity_mach[i], Xtest[i]) for i in 1:3]
            mse_outcome = mean(reduce(vcat, (oos_outcome_preds[i] .- true_conmean[i]).^2 for i in 1:3))
            mse_propensity = mean(reduce(vcat, (oos_propensity_preds[i] .- true_prob[i]).^2 for i in 1:3))

            # Compute one-step estimates using models
            pr_indices = [(1, 2), (2, 3), (3, 1)]
            μ_indices = [(1, 3), (2, 1), (3, 2)]
            prA = reduce(vcat, [safe_predict(propensity_mach[j], X[i], 0.02, 0.98) for (i, j) in pr_indices])
            μ = reduce(vcat, [safe_predict(outcome_mach[j], XA[i], miny[i], maxy[i]) for (i, j) in μ_indices])
            μ1 = reduce(vcat, [safe_predict(outcome_mach[j], XA_A1[i], miny[i], maxy[i]) for (i, j) in μ_indices])
            μ0 = reduce(vcat, [safe_predict(outcome_mach[j], XA_A0[i], miny[i], maxy[i]) for (i, j) in μ_indices])

            # Cross-fit EIF
            A_full = reduce(vcat, A)
            y_full = reduce(vcat, y)
            eif = μ1 .- μ0 .+ ((A_full ./ prA) .- ((1 .- A_full) ./ (1 .- prA))) .* (y_full .- μ)

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