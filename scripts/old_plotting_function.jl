function generate_plots(df_raw, str, eff_bound, models)
    ns = unique(df_raw.n)
    df_raw[!, "upper"] = df_raw.ose .+ 1.96 .* sqrt.(df_raw.ose_var)
    df_raw[!, "lower"] = df_raw.ose .- 1.96 .* sqrt.(df_raw.ose_var)

    df = @chain df_raw begin
        @groupby(:n, :model_name)
        @combine(:mean_mse_outcome = mean(:mse_outcome), 
                :mean_mse_propensity = mean(:mse_propensity), 
                :mean_bias = mean(:ose) .- mean(:true_ate), 
                :mean_ose_var = mean(:ose_var),
                :mc_var = var(:ose),
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
    df[!, "scaled_mse"] = df.n .* ((df.mean_bias .^ 2) .+ df.mc_var) ./ eff_bound

    # keep only specified models
    df = filter(row -> row.model in models, df)

    # Set up common plot elements
    line_styles = [:solid, :dash, :dot, :dashdot]
    markers = [:circle, :rect, :diamond, :utriangle]
    smoothness_levels = unique(df.smoothness)
    color_labels = unique(df.model)
    color_elements = [LineElement(color=Makie.wong_colors()[mod1(i, length(Makie.wong_colors()))], linewidth=3) for i in eachindex(color_labels)]
    smooth_labels = string.(smoothness_levels)
    smooth_elements = [LineElement(color=:black, linewidth=3, linestyle=line_styles[mod1(i, length(line_styles))], marker=markers[mod1(i, length(markers))]) for i in eachindex(smooth_labels)]

    # Figure 1
    fig = Figure(; size=(1700, 450))

    ymaxout = maximum(df.mean_mse_outcome) * 1.05
    ymaxprop = maximum(df.mean_mse_propensity) * 1.05
    for (smooth_idx, smooth_level) in enumerate(smoothness_levels)
        df_smooth = filter(row -> row.smoothness == smooth_level, df)
        smooth_style = line_styles[mod1(smooth_idx, length(line_styles))]
        smooth_marker = markers[mod1(smooth_idx, length(markers))]
        template_smooth = data(df_smooth) * visual(Lines, markersize=8, linestyle=smooth_style, marker=smooth_marker) * visual(ScatterLines, markersize=8, marker=smooth_marker)
        col_offset = 2 * (smooth_idx - 1)

        Label(fig[0, col_offset + 1:col_offset + 2], "Smoothness: $(smooth_level)", fontsize=18)

        p1 = template_smooth * 
            mapping(:n, :mean_mse_outcome, color=:model, linestyle=:smoothness, marker=:smoothness)
        # set y-limits manually based on data
        draw!(fig[1, col_offset + 1], p1, axis=(xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = "Outcome model", limits=(nothing, (0.0, ymaxout))))

        p2 = template_smooth * 
            mapping(:n, :mean_mse_propensity, color=:model, linestyle=:smoothness, marker=:smoothness)
        draw!(fig[1, col_offset + 2], p2, axis=(xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = "Propensity model", limits=(nothing, (0.0, ymaxprop))))
    end

    Legend(fig[1, 2 * length(smoothness_levels) + 1], color_elements, color_labels, "Model")
    #Legend(fig[2, 2 * length(smoothness_levels) + 1], smooth_elements, smooth_labels, "Smoothness pattern")
    save(plotsdir(str*"MSE.png"), fig)

    # Figure 2    
    fig = Figure(; size=(1300, 620))

    for (smooth_idx, smooth_level) in enumerate(smoothness_levels)
        df_smooth = filter(row -> row.smoothness == smooth_level, df)
        smooth_style = line_styles[mod1(smooth_idx, length(line_styles))]
        smooth_marker = markers[mod1(smooth_idx, length(markers))]
        template_smooth = data(df_smooth) * visual(Lines, markersize=8, linestyle=smooth_style, marker=smooth_marker) * visual(ScatterLines, markersize=8, marker=smooth_marker)
        col_offset = 2 * (smooth_idx - 1)

        Label(fig[0, col_offset + 1:col_offset + 2], "Smoothness: $(smooth_level)", fontsize=18)

        p1 = template_smooth * 
            mapping(:n, :mean_bias, color=:model)
        draw!(fig[1, col_offset + 1], p1, axis=create_axis_ose(ns, "Bias"))

        p2 = template_smooth * 
            mapping(:n, :scaled_bias, color=:model)
        draw!(fig[1, col_offset + 2], p2, axis=create_axis_ose(ns, "Scaled bias"))

        p3 = template_smooth * 
            mapping(:n, :scaled_mse, color=:model)
        draw!(fig[2, col_offset + 1], p3, axis=create_axis_ose(ns, "Scaled MSE"))

        p4 = (template_smooth * mapping(:n, :coverage, color=:model)) + 
            (visual(HLines) * mapping([0.95]))
        draw!(fig[2, col_offset + 2], p4, axis=(aspect=1, xticks=ns, yticks = [0.0, 0.5, 0.95], xlabel = "Sample size", ylabel="Coverage", limits=(nothing, (0.0, 1.0))))
    end

    Legend(fig[1, 2 * length(smoothness_levels) + 1], color_elements, color_labels, "Model")
    #Legend(fig[2, 2 * length(smoothness_levels) + 1], smooth_elements, smooth_labels, "Smoothness pattern")
    save(plotsdir(str*"onestep.png"), fig)

    # Figure 3: CATE per smoothness (panels side-by-side with shared legend)
    fig = Figure(; size=(1300, 360))
    for (smooth_idx, smooth_level) in enumerate(smoothness_levels)
        df_smooth = filter(row -> row.smoothness == smooth_level, df)
        smooth_style = line_styles[mod1(smooth_idx, length(line_styles))]
        smooth_marker = markers[mod1(smooth_idx, length(markers))]
        template_smooth = data(df_smooth) * visual(Lines, markersize=8, linestyle=smooth_style, marker=smooth_marker) * visual(ScatterLines, markersize=8, marker=smooth_marker)

        col = smooth_idx
        Label(fig[0, col], "Smoothness: $(smooth_level)", fontsize=16)
        p = template_smooth * mapping(:n, :mean_cate_mse, color=:model, linestyle=:smoothness, marker=:smoothness)
        draw!(fig[1, col], p, axis=(aspect=1, xticks=ns, xlabel = "Sample size", ylabel="Out-of-sample MSE", title = "CATE Estimate"))
    end
    Legend(fig[1, length(smoothness_levels) + 1], color_elements, color_labels, "Model")
    save(plotsdir(str*"cate.png"), fig)

    # Figure 4: Timing per smoothness (panels similar to Figure 1/2)
    fig = Figure(; size=(1300, 420))
    for (smooth_idx, smooth_level) in enumerate(smoothness_levels)
        df_smooth = filter(row -> row.smoothness == smooth_level, df)
        smooth_style = line_styles[mod1(smooth_idx, length(line_styles))]
        smooth_marker = markers[mod1(smooth_idx, length(markers))]
        template_smooth = data(df_smooth) * visual(Lines, markersize=8, linestyle=smooth_style, marker=smooth_marker) * visual(ScatterLines, markersize=8, marker=smooth_marker)
        col_offset = 2 * (smooth_idx - 1)

        Label(fig[0, col_offset + 1:col_offset + 2], "Smoothness: $(smooth_level)", fontsize=18)

        p1 = template_smooth * mapping(:n, :mean_time_outcome, color=:model)
        draw!(fig[1, col_offset + 1], p1, axis=create_axis_time(ns, "Outcome model"))

        p2 = template_smooth * mapping(:n, :mean_time_propensity, color=:model)
        draw!(fig[1, col_offset + 2], p2, axis=create_axis_time(ns, "Propensity model"))
    end

    Legend(fig[1, 2 * length(smoothness_levels) + 1], color_elements, color_labels, "Model")
    save(plotsdir(str*"time.png"), fig)
end