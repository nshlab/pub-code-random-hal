
using Pkg
Pkg.activate("RandomHAL.jl/sims")

using RandomHAL
using CausalTables
using Distributions
using MLJ
using Logging
using Plots
using Distributed
using JLD2

import LogExpFunctions: logistic
disable_logging(Logging.Info)
disable_logging(Logging.Warn)

dgp = @dgp(
        X2 ~ Beta(2, 3),
        X3 ~ Beta(3, 2),
        X4 ~ Beta(3, 3),
        A ~ (@. Bernoulli(logistic((X2 + X2^2 + X3 + X3^2 + X4 + X4^2) - 2.5))),
        Y ~ (@. Normal(A + sqrt(10*X3*X4) + sqrt(10 * X2) + sqrt(10 * X3) + sqrt(10*X4) + 5, 1.0))
    )

scm = StructuralCausalModel(dgp2, :A, :Y)
truth = ate(scm, samples = 10^7)
ct = rand(scm, 10^7)
ct_A0 = intervene(ct, treat_none)
ct_A1 = intervene(ct, treat_all)
prop = propensity(scm, ct, :A)
true_eif = (conmean(scm, ct_A1, :Y) - conmean(scm, ct_A0, :Y)) .+ (((ct.data.A .== 1) ./ prop) .- ((ct.data.A .== 0) ./ prop)) .* (ct.data.Y .- conmean(scm, ct, :Y))
eff_bound = var(true_eif)

# Combination that worked: simulate2, dgp, no penalty exemption

function simulate(n, modellist, iters)

    mse = [Vector{Float64}(undef, iters) for _ in modellist]
    bias = [Vector{Vector{Float64}}(undef, iters) for _ in modellist]
    props = [Vector{Vector{Float64}}(undef, iters) for _ in modellist]
    plugins = [Vector{Float64}(undef, iters) for _ in modellist]
    oses = [Vector{Float64}(undef, iters) for _ in modellist]
    oses_var = [Vector{Float64}(undef, iters) for _ in modellist]


    Threads.@threads for i in 1:iters
        println("Iteration $i, Thread $(Threads.threadid())")
        # Train models
        ct = rand(scm, n)
        XA = responseparents(ct)
        X = treatmentparents(ct)
        A = treatmentmatrix(ct)[:,1]
        y = responsematrix(ct)[:, 1]

        cttest = rand(scm, n)
        XAtest = responseparents(cttest)
        Xtest = responseparents(cttest)
        Atest = treatmentmatrix(ct)[:,1]
        ytest = responsematrix(cttest)[:, 1]
        true_conmean = conmean(scm, cttest, :Y)

        cttest_A1 = intervene(cttest, treat_all)
        XAtest_A1 = responseparents(cttest_A1)
        cttest_A0 = intervene(cttest, treat_none)
        XAtest_A0 = responseparents(cttest_A0)

        #cttest_intervene = intervene(cttest, intervention)
        #Xtest_intervene = responseparents(cttest_intervene)
        for (j, model) in enumerate(modellist)
                mach = machine(model[1], XA, y) |> fit!
                preds = MLJ.predict_mean(mach, XAtest)
                mse[j][i] = mean((preds .- ytest).^2)
                bias[j][i] = preds .- true_conmean

                diff = MLJ.predict_mean(mach, XAtest_A1) - MLJ.predict_mean(mach, XAtest_A0)
                plugin = mean(diff)
                plugins[j][i] = plugin

                mach2 = machine(model[2], X, A) |> fit!
                # TODO: Truncate this
                propensity = MLJ.predict_mean(mach2, Xtest)
                props[j][i] = propensity .- conmean(scm, cttest, :A)

                resid = (((Atest .== 1) ./ propensity) .- ((Atest .== 0) ./ (1 .- propensity))) .* (ytest .- MLJ.predict_mean(mach, XAtest))
                ose = plugin + mean(resid)
                oses[j][i] = ose
                σ2 = var(diff .+ resid)
                oses_var[j][i] = σ2

        end
    end
    return (mse = mse, bias = bias, plugin = plugins, ose = oses, props = props)
end

function simulate2(n, modellist, iters)

        mse = [Vector{Float64}(undef, iters) for _ in modellist]
        #bias = [Vector{Vector{Float64}}(undef, iters) for _ in modellist]
        #props = [Vector{Vector{Float64}}(undef, iters) for _ in modellist]
        plugins = [Vector{Float64}(undef, iters) for _ in modellist]
        oses = [Vector{Float64}(undef, iters) for _ in modellist]
        oses_var = [Vector{Float64}(undef, iters) for _ in modellist]
    
        Threads.@threads for i in 1:iters
            println("Iteration $i, Thread $(Threads.threadid())")
            # Train models
            ct = rand(scm, n)
            XA = responseparents(ct)
            X = treatmentparents(ct)
            A = treatmentmatrix(ct)[:,1]
            y = responsematrix(ct)[:, 1]
            #true_conmean = conmean(scm, ct, :Y)


            ct_A1 = intervene(ct, treat_all)
            XA_A1 = responseparents(ct_A1)
            ct_A0 = intervene(ct, treat_none)
            XA_A0 = responseparents(ct_A0)
    
            for (j, model) in enumerate(modellist)
                    mach = machine(model[1], XA, y) |> fit!
                    preds = MLJ.predict(mach, XA)
                    mse[j][i] = mean((preds .- y).^2)
                    #bias[j][i] = preds .- true_conmean
    
                    diff = MLJ.predict(mach, XA_A1) - MLJ.predict(mach, XA_A0)
                    plugin = mean(diff)
                    plugins[j][i] = plugin
    
                    mach2 = machine(model[2], X, A) |> fit!
                    propensity = MLJ.predict(mach2, X)
                    #props[j][i] = propensity .- conmean(scm, ct, :A)
    
                    resid = (((A .== 1) ./ propensity) .- ((A .== 0) ./ (1 .- propensity))) .* (y .- MLJ.predict(mach, XA))
                    ose = plugin + mean(resid)
                    oses[j][i] = ose
                    σ2 = var(diff .+ resid)
                    oses_var[j][i] = σ2
    
            end
        end
        return (mse = mse, plugin = plugins, ose = oses, ose_var = oses_var)
end

function simulate3(n, modellist, iters)

        mse = [Vector{Float64}(undef, iters) for _ in modellist]
        plugins = [Vector{Float64}(undef, iters) for _ in modellist]
        oses = [Vector{Float64}(undef, iters) for _ in modellist]
        oses_var = [Vector{Float64}(undef, iters) for _ in modellist]
    
        Threads.@threads for i in 1:iters
                println("Iteration $i, Thread $(Threads.threadid())")
                # Train models
                ct = [rand(scm, Int(round(n/2))), rand(scm, Int(round(n/2)))]
                XA = responseparents.(ct)
                X = treatmentparents.(ct)
                A = [treatmentmatrix(c)[:,1] for c in ct]
                y = [responsematrix(c)[:, 1] for c in ct]
                true_conmean = vcat([conmean(scm, c, :Y) for c in ct]...)

                ctA1 = [intervene(c, treat_all) for c in ct]
                XA1 = [responseparents(c) for c in ctA1]
                ctA0 = [intervene(c, treat_none) for c in ct]
                XA0 = [responseparents(c) for c in ctA0]
    
            for (j, model) in enumerate(modellist)
                mach = [fit!(machine(model[1], XA[2], y[2])), fit!(machine(model[1], XA[1], y[1]))]
                preds = vcat(MLJ.predict_mean(mach[1], XA[1]), MLJ.predict_mean(mach[2], XA[2]))
                mse[j][i] = mean((preds .- true_conmean).^2)

                dif = vcat(MLJ.predict_mean(mach[1], XA1[1]) - MLJ.predict_mean(mach[1], XA0[1]),
                        MLJ.predict_mean(mach[2], XA1[2]) - MLJ.predict_mean(mach[2], XA0[2]))
                plugin = mean(dif)
                plugins[j][i] = plugin

                mach2 = [fit!(machine(model[2], X[2], A[2])), fit!(machine(model[2], X[1], A[1]))]

                # TODO: Truncate this
                prop = vcat(MLJ.predict_mean(mach2[1], X[1]), MLJ.predict_mean(mach2[2], X[2]))
                Av = vcat(A...)
                yv = vcat(y...)
                resid = (((Av .== 1) ./ prop) .- ((Av .== 0) ./ (1 .- prop))) .* (yv .- preds)
                ose = plugin + mean(resid)
                oses[j][i] = ose
                σ2 = var(dif .+ resid)
                oses_var[j][i] = σ2
    
            end
        end
        return (mse = mse, plugin = plugins, ose = oses, ose_var = oses_var)
    end

function simulate4(n, modellist, iters)

        mse = [Vector{Float64}(undef, iters) for _ in modellist]
        bias = [Vector{Vector{Float64}}(undef, iters) for _ in modellist]
        props = [Vector{Vector{Float64}}(undef, iters) for _ in modellist]
        plugins = [Vector{Float64}(undef, iters) for _ in modellist]
        oses = [Vector{Float64}(undef, iters) for _ in modellist]
        oses_var = [Vector{Float64}(undef, iters) for _ in modellist]
    
        Threads.@threads for i in 1:iters
            println("Iteration $i, Thread $(Threads.threadid())")
            # Train models
            ct = rand(scm, n)
            XA = responseparents(ct)
            X = treatmentparents(ct)
            A = treatmentmatrix(ct)[:,1]
            y = responsematrix(ct)[:, 1]

            cttest = rand(scm, n)
            XAtest = responseparents(cttest)
            ytest = responsematrix(cttest)[:, 1]
            true_conmean = conmean(scm, cttest, :Y)

            ct_A1 = intervene(ct, treat_all)
            XA_A1 = responseparents(ct_A1)
            ct_A0 = intervene(ct, treat_none)
            XA_A0 = responseparents(ct_A0)
    
            for (j, model) in enumerate(modellist)
                    mach = machine(model[1], XA, y) |> fit!
                    preds = MLJ.predict(mach, XAtest)
                    mse[j][i] = mean((preds .- ytest).^2)
                    bias[j][i] = preds .- true_conmean
    
                    dif = MLJ.predict(mach, XA_A1) - MLJ.predict(mach, XA_A0)
                    plugin = mean(dif)
                    plugins[j][i] = plugin
    
                    mach2 = machine(model[2], X, A) |> fit!
                    propensity = MLJ.predict(mach2, X)
                    props[j][i] = propensity .- conmean(scm, ct, :A)
    
                    resid = (((A .== 1) ./ propensity) .- ((A .== 0) ./ (1 .- propensity))) .* (y .- MLJ.predict(mach, XA))
                    ose = plugin + mean(resid)
                    oses[j][i] = ose
                    σ2 = var(dif .+ resid)
                    oses_var[j][i] = σ2
    
            end
        end
        return (mse = mse, plugin = plugins, bias = bias, props = props, ose = oses, ose_var = oses_var)
end

ns = (10:10:20).^2
iters = 20

@time for n in ns
        println("Sample Size: $(n)")
        r = simulate4(n, [(HALRegressor(0, 100, 5), HALBinaryClassifier(0)), 
                (RandomHALRegressor(0, round(n^(5/4)), 100, 5), RandomHALBinaryClassifier(0, round(n^(5/4))))], iters)
        JLD2.save("experiment-25-03-06_n=$(n).jld2", Dict("result" => r))
end

result = [JLD2.load("experiment-25-03-06_n=$(n).jld2")["result"] for n in ns]
### Prediction Plots ###
# MSE
halmse = [mean(r.mse[1]) for r in result]
rhalmse = [mean(r.mse[2]) for r in result]

plot(ns, [halmse rhalmse], label = ["HAL" "RandomHAL"], 
        xlabel = "Sample Size", ylabel = "Mean-Squared Error",
        xticks = ns, size = (400, 300))
hline!([0], label = "")
result[1]
halpropmse = [mean([mean(rp.^2) for rp in r.props[1]]) for r in result]
rhalpropmse = [mean([mean(rp.^2) for rp in r.props[2]]) for r in result]

plot(ns, [halpropmse rhalpropmse], label = ["HAL" "RandomHAL"], 
        xlabel = "Sample Size", ylabel = "Mean-Squared Error",
        xticks = ns, size = (400, 300))
hline!([0], label = "")

# Bias
halbias = [mean(mean(b) for b in r.bias[1]) for r in result]
rhalbias = [mean(mean(b) for b in r.bias[2]) for r in result]

plot(string.(ns), [halbias rhalbias], label = ["HAL" "RandomHAL"], 
        xlabel = "n", ylabel = "Mean Bias")
hline!([0], label = "")

# Prop Bias
#halprop = [mean(mean(b) for b in r.props[1]) for r in result]
#rhalprop = [mean(mean(b) for b in r.props[2]) for r in result]

#plot(string.(ns), [halprop rhalprop], label = ["HAL" "RandomHAL"], 
#        xlabel = "n", ylabel = "Mean Bias")
#hline!([0], label = "")

# Scaled Bias
#nhalbias = ns.^(1/4) .* halbias
#nrhalbias = ns.^(1/4) .* rhalbias

#plot(string.(ns), [nhalbias nrhalbias], label = ["HAL" "RandomHAL"], 
#        xlabel = "Sample Size", ylabel = "Quarter-root Scaled Bias")
#hline!([0], label = "")

### ATE Functional Plots ###
# Compute a simple functional, treatment-specific mean, without undersmoothing
# Two figures: pointwise bias (with band over the simulation replicates) + grid of 2 plots (scaled bias of the functional, scaled MSE of the functional going to efficiency bound)

hal_ate_bias = [mean(r.plugin[1] .- truth.μ) for r in result]
rhal_ate_bias = [mean(r.plugin[2] .- truth.μ) for r in result]

hal_ate_var = [var(r.plugin[1]) for r in result]
rhal_ate_var = [var(r.plugin[2]) for r in result]
ci = 1.96 .* sqrt.([hal_ate_var rhal_ate_var])

hal_ate_smse = ns .* ([mean((r.plugin[1] .- 1).^2) for r in result] .+ hal_ate_var)
rhal_ate_smse = ns .* ([mean((r.plugin[2] .- 1).^2)  for r in result] .+ rhal_ate_var)

p1 = plot(ns, abs.([hal_ate_bias rhal_ate_bias]) ./ truth.μ, label = ["HAL" "RandomHAL"], #yerror = ci, 
        markerstrokecolor = :auto, 
        xlabel = "Sample Size", ylabel = "ATE Bias", xticks = ns);
hline!([0], label = "");

p2 = plot(ns, sqrt.(ns) .* abs.([hal_ate_bias rhal_ate_bias]), label = ["HAL" "RandomHAL"],# yerror = ci .* ns, 
        markerstrokecolor = :auto, 
        xlabel = "Sample Size", ylabel = "√n-scaled ATE Bias", xticks = ns, legend = false);
hline!([0], label = "");

p3 = plot(ns, [hal_ate_smse rhal_ate_smse], label = ["HAL" "RandomHAL"], markerstrokecolor = :auto, 
        xlabel = "Sample Size", ylabel = "n-scaled ATE MSE", xticks = ns, legend = false);
hline!([eff_bound], label = "");

plot(p1, p2, p3, layout = (3, 1), size = (400, 800), left_margin=8Plots.mm)
savefig("plugin_25-03-06-cluster.pdf")

# Same but for ose

hal_ate = [mean(r.ose[1]) for r in result]
rhal_ate = [mean(r.ose[2]) for r in result]
hal_ate_bias = hal_ate .- truth.μ
rhal_ate_bias = rhal_ate .- truth.μ

hal_ate_var = [var(r.ose[1]) for r in result]
rhal_ate_var = [var(r.ose[2]) for r in result]

ci = 1.96 .* sqrt.([hal_ate_var rhal_ate_var])

# What I was doing before
#hal_ate_bias2 = [mean((r.ose[1] .- truth.μ))^2 for r in result]
#rhal_ate_bias2 = [mean((r.ose[2] .- truth.μ).^2)  for r in result]
#hal_ate_smse = ns .* (hal_ate_bias2 .+ hal_ate_var)
#rhal_ate_smse = ns .* (rhal_ate_bias2 .+ rhal_ate_var)

hal_ate_smse = ns .* (hal_ate_bias.^2 .+ hal_ate_var)
rhal_ate_smse = ns .* (rhal_ate_bias.^2 .+ rhal_ate_var)


p1 = plot(ns, abs.([hal_ate_bias rhal_ate_bias]), label = ["HAL" "RandomHAL"], #yerror = ci, 
        markerstrokecolor = :auto, 
        xlabel = "Sample Size", ylabel = "ATE Bias", xticks = ns);
hline!([0], label = "");

p2 = plot(ns, sqrt.(ns) .* abs.([hal_ate_bias rhal_ate_bias]), label = ["HAL" "RandomHAL"],# yerror = ci .* ns, 
        markerstrokecolor = :auto, 
        xlabel = "Sample Size", ylabel = "√n-scaled ATE Bias", xticks = ns, legend = false);
hline!([0], label = "");


# TODO: Try instead scaling the eff_bound by n and plotting this against unscaled MSE
p3 = plot(ns, [hal_ate_smse rhal_ate_smse], label = ["HAL" "RandomHAL"], markerstrokecolor = :auto, 
        xlabel = "Sample Size", ylabel = "n-scaled ATE MSE", xticks = ns, legend = false);
hline!([eff_bound], label = "");

# CI
#hal_upper = [1.96 * sqrt.(r.ose_var[1]) for r in result] ./ sqrt.(ns)
#rhal_width = [1.96 * sqrt.(r.ose_var[2]) for r in result] ./ sqrt.(ns)
#hal_upper = [r.ose[1] for r in result] .+ hal_width
#hal_lower = [r.ose[1] for r in result] .- hal_width
#rhal_upper = [r.ose[2] for r in result] .+ rhal_width
#rhal_lower = [r.ose[2] for r in result] .- rhal_width

#hal_ate_ci = [mean((hal_upper[i] .> 1) .| (hal_lower[i] .> 1)) for i in 1:length(hal_upper)]
#rhal_ate_ci = [mean((rhal_upper[i] .> 1) .| (rhal_lower[i] .> 1)) for i in 1:length(rhal_upper)]


plot(p1, p2, p3, layout = (3, 1), size = (400, 800), left_margin=8Plots.mm)
savefig("ose_25-03-06-cluster.pdf")