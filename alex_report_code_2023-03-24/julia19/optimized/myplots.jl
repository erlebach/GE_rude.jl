
# using Colors

function plot_data!(k, plots, target_label, protocol_label, sols, named_params)
    #For each protocol (target), we compute σ12, N1, and N2 from the UDE and Giesekus solutions
    Plots.default(
        # fontfamily = :Helvetica,
        guidefont=5,
        legendfontsize=4, # = (5,  :Helvetica),
        titlefontsize=5,
        xtickfontsize=4,
        ytickfontsize=4,
    )

    base, pre, post = sols
    println("k, target, protocol: $k, $target_label, $protocol_label")

    if (k == 1 && target_label == "N1")   # arbitrary plot
        legend = :topright
    else
        legend = false
    end

    # sol can be one of (base, pre, post)
    σ12fct = sol -> sol[4, :]
    N1fct = sol -> sol[1, :] .- sol[2, :]
    N2fct = sol -> sol[2, :] .- sol[3, :]

    # Plot the data
    plot(xlabel="Time", xlims=tspan, legend=legend, background_color_legend=colorant"rgba(255,255,255,0.7)")

    # When the blue line is not seen, it is because the red and blue lines are superimposed. This means that the converged
    # state is near the initial state. This might happen where you run only a few epochs of the neural network. 

    plotfct = function (sols, target_base, target_pre, target_post)
        base, pre, post = sols
        # mss = 1.25  # Marker size
        plot!(pre.t, target_pre, c=:blue, ls=:dash, lw=3.0 * 0.5, title="$target_label, v21=$protocol_label", label="UDE-pre")
        plot!(post.t, target_post, c=:red, lw=1.5 * 0.5, label="UDE-post")
        scatter!(base.t[1:2:end], target_base[1:2:end], m=:o, c=:black, ms=1.0, label="Giesekus")
        # return return value of plot!
        plot!(base.t[1:2:end], target_base[1:2:end], lw=0.50 * 0.5, c=:black)
    end

    # Create another losure to simplify these conditionals
    # plots = []
    println("===> k= $k")
    if startswith(target_label, "σ12")
        plot_σ12 = plotfct(sols, σ12fct(base), σ12fct(pre), σ12fct(post))
        # println("protocol: $(protocol_label)")
        push!(plots, plot_σ12)
    elseif startswith(target_label, "N1")
        plot_N1 = plotfct(sols, N1fct(base), N1fct(pre), N1fct(post))
        push!(plots, plot_N1)
    elseif startswith(target_label, "N2")
        plot_N2 = plotfct(sols, N2fct(base), N2fct(pre), N2fct(post))
        push!(plots, plot_N2)
    elseif startswith(target_label, "ηE")
        println("ηE not implemented")
    end
    halt = false  # not used
    params = ", "
    ks = keys(named_params)
    for (i,v) in enumerate(named_params)
        params *= string(ks[i])
        params *= "=$v, "
    end
    plot(plots..., plot_title="(Protocols, Targets)"*params, plot_titlefontsize=12, layout=(4, 4))
    savefig("all_plots.pdf")
    return plots, halt
end


#----------------------------------------------------------------------
function plot_solution(θ0, θi, p_system, σ0, p_giesekus; dct)
    println("plot_solution: ", dct |> keys)
    # Define the simple shear deformation protocol
    v11(t) = 0
    v12(t) = 0
    v13(t) = 0
    v22(t) = 0
    v23(t) = 0
    v31(t) = 0
    v32(t) = 0
    v33(t) = 0

    # Parameters of the linear response (η0,τ)
    p_system = Float32[1, 1]

    θ0 = [θ0; p_system]  # create single column vector
    θi = [θi; p_system]
    #@show θ0, θi
    ω = dct[:ω]
    γ = dct[:γ]

    # Test the UDE on a new condition (different ω, different γ)
    v21_1(t) = 2 * cos(3 * ω * t / 4)
    v21_2(t) = 2 * cos(ω * t)
    v21_3(t) = 2 * cos(ω * t)  # same as v21_2
    v21_4(t) = 1.5f0
    gradv_1 = [v11, v12, v13, v21_1, v22, v23, v31, v32, v33]
    gradv_2 = [v11, v12, v13, v21_2, v22, v23, v31, v32, v33]
    gradv_3 = [v11, v12, v13, v21_3, v22, v23, v31, v32, v33]
    gradv_4 = [v11, v12, v13, v21_4, v22, v23, v31, v32, v33]
    protocols = [gradv_1, gradv_2, gradv_3, gradv_4]
    targets = ["σ12a", "N1", "N2", "σ12b"]
    target_titles = ["2cos(3ωt/4)", "2cos(ωt)", "2cos(ωt)", "1.5"]

    #tspan = (0.0f0, dct[:T])
    tspan = (0.0f0, 2.0f0 * dct[:T])  # test extrapolation
    #println("plot_solution: tspan: ", tspan)

    tdnn_traces = []
    tdnn_coefs = []
    tdnn_Fs = []
    plots = []
    halt = false
    for k = range(1, length(protocols), step=1)  # ORIGINAL
        # Solve the Giesekus model (σ11, σ22, σ33, σ12, σ13, σ23)
        dudt!(du, u, p, t) = dudt_giesekus!(du, u, p, t, protocols[k])
        prob_giesekus = ODEProblem(dudt!, σ0, tspan, p_giesekus)
        sol_giesekus = solve(prob_giesekus, Tsit5(), saveat=dct[:saveat])

        dudt_ude!(du, u, p, t) = dudt_univ!(du, u, p, t, protocols[k], dct)

        # Solve the UDE pre-training  (use parameters θ0)
        prob_univ_pre = ODEProblem(dudt_ude!, σ0, tspan, θ0)
        sol_ude_pre = solve(prob_univ_pre, Tsit5(), abstol=1e-7, reltol=1e-6, saveat=dct[:saveat])

        # Solve the UDE post-training (use parameters θi)
        prob_univ_post = ODEProblem(dudt_ude!, σ0, tspan, θi)
        println("plot_solution, dct[:final_plot]: ", dct[:final_plot])

        if dct[:final_plot] == true
            dct[:captureG] = true
        end
        sol_ude_post = solve(prob_univ_post, abstol=1e-7, reltol=1e-6, saveat=dct[:saveat])
        dct[:captureG] = false

        # Perhaps I should write this every time this function is called, and simply overwrite the previous save
        # In that case, remove the next line. Perhaps rename the file with the iteraiton and/or protocol?
        if dct[:final_plot] == true
            # Write binary information with npzwrite: a single array 
            NPZ.npzwrite("tdnn_traces_k=$k.npz", reduce(hcat, dct[:tdnn_traces]))
            NPZ.npzwrite("tdnn_Fs_k=$k.npz", reduce(hcat, dct[:tdnn_Fs]))
            NPZ.npzwrite("tdnn_coefs_k=$k.npz", reduce(hcat, dct[:tdnn_coefs]))
            # npzwrite cannot write strings
            #@save "targets.bson" targets
            empty!(tdnn_traces)
            empty!(tdnn_Fs)
            empty!(tdnn_coefs)
        end

        println("==> call to plot_data: targets[$k]: $(targets[k]), $(target_titles[k])")
        plot_data!(plots, targets[k], target_titles[k], sol_ude_pre, sol_ude_post, sol_giesekus, tspan)
    end

    println("return from plot_solution, len(plots): ", length(plots))
    println("return from plot_solution, halt: ", halt)
    return plots, halt
end

#----------------------------------------------------------------------
# Implementation without dictionary. The function should be self contained
#function my_plot_solution(θ0, θi, p_system, σ0, p_giesekus; dct)
function my_plot_solution(θ0, θi, protocols, labels, fcts, named_params)
    # Define the simple shear deformation protocol
    # Parameters of the linear response (η0,τ)
    p_system = p_giesekus

    target_labels = labels.target
    protocol_labels = labels.protocol

    plots = Any[]

    # The result only changes with the protocols.
    # No change with respect to target
    for k = range(1, length(protocols), step=1)  # ORIGINAL
        println(protocol_labels)
        println("==> my_plot_solution, k= $k ==> length(protocols): $(length(protocols))")
        # @show length(protocols)
        protocol_label = protocol_labels[k]
        sol_giesekus = fcts.base_model(k)
        # println("==> len θ0: ", length(θ0))
        # println("==> len θi: ", length(θi))
        sol_ude_pre = fcts.ude_model(k, θ0)
        sol_ude_post = fcts.ude_model(k, θi)
        sols = (base=sol_giesekus, pre=sol_ude_pre, post=sol_ude_post)

        for target_label in target_labels
            plots, halt = plot_data!(k, plots, target_label, protocol_label, sols, named_params)
        end
    end

    halt = false
    return plots, halt
end

#----------------------------------------------------------------------

