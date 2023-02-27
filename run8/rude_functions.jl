# The functions in this file do not access a global dictionary. 
# Ideally, they should not access global variables (for portability). I am not sure that is the case. 

function dudt_giesekus!(du, u, p, t, gradv)
        # # Update in place
        # du[1] = 0.
        # du[2] = 0.
        # du[3] = 0.
        # du[4] = 0.
        # du[5] = 0.
        # du[6] = 0.
        # return du


    # Destructure the parameters
    η0 = p[1]
    τ = p[2]
    α = p[3]

    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u

    # Specify the velocity gradient tensor
    # ∇v  ∂vᵢ / ∂xⱼ v12: partial derivative of the 1st component 
    # of v(x1, x2, x3) with respect to the 3rd component of x
    v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv    

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)
    γd22 = 2*v22(t)
    γd33 = 2*v33(t)
    γd12 = v12(t) + v21(t)
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)
    ω12 = v12(t) - v21(t)
    ω13 = v13(t) - v31(t)
    ω23 = v23(t) - v32(t)

    # Define F for the Giesekus model
    F11 = -τ*(σ11*γd11 + σ12*γd12 + σ13*γd13) + (α*τ/η0)*(σ11^2 + σ12^2 + σ13^2)
    F22 = -τ*(σ12*γd12 + σ22*γd22 + σ23*γd23) + (α*τ/η0)*(σ12^2 + σ22^2 + σ23^2)
    F33 = -τ*(σ13*γd13 + σ23*γd23 + σ33*γd33) + (α*τ/η0)*(σ13^2 + σ23^2 + σ33^2)
    F12 = (-τ*(σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23)/2
    + (α*τ/η0)*(σ11*σ12 + σ12*σ22 + σ13*σ23))
    F13 = (-τ*(σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33)/2
    + (α*τ/η0)*(σ11*σ13 + σ12*σ23 + σ13*σ33))
    F23 = (-τ*(σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33)/2
    + (α*τ/η0)*(σ12*σ13 + σ22*σ23 + σ23*σ33))

    ##
    # The model differential equations
    du[1] = η0*γd11/τ - σ11/τ - (ω12*σ12 + ω13*σ13) - F11/τ
    du[2] = η0*γd22/τ - σ22/τ - (ω23*σ23 - ω12*σ12) - F22/τ
    du[3] = η0*γd33/τ - σ33/τ + (ω13*σ13 + ω23*σ23) - F33/τ
    du[4] = η0*γd12/τ - σ12/τ - (ω12*σ22 + ω13*σ23 - σ11*ω12 + σ13*ω23)/2 - F12/τ
    du[5] = η0*γd13/τ - σ13/τ - (ω12*σ23 + ω13*σ33 - σ11*ω13 - σ12*ω23)/2 - F13/τ
    du[6] = η0*γd23/τ - σ23/τ - (ω23*σ33 - ω12*σ13 - σ12*ω13 - σ22*ω23)/2 - F23/τ
    ##

    #= Update in place
    du[1] = 0.
    du[2] = 0.
    du[3] = 0.
    du[4] = 0.
    du[5] = 0.
    du[6] = 0.
    return  # REMOVE THIS AND ABVOE du[i] = 0 ONCE DEBUGGED
    =#
end

function tbnn(σ,γd,model_weights, model_univ)
	# Remove next three lines once Debugging complete)
    #model_inputs = [0.;0.;0.;0.;0.;0.;0.;0.;0.]
    #g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(model_inputs, model_weights)  # Will this create the warning ?
    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE. No warning generated. 

    # Tensor basis neural network (TBNN)
    # Unpack the inputs
    σ11,σ22,σ33,σ12,σ13,σ23 = σ
    γd11,γd22,γd33,γd12,γd13,γd23 = γd

    # Compute elements of the tensor basis
    # T1 = I, T2 = σ, T3 = γd
    # T4 = σ⋅σ
    T4_11 = σ11^2 + σ12^2 + σ13^2
    T4_22 = σ12^2 + σ22^2 + σ23^2
    T4_33 = σ13^2 + σ23^2 + σ33^2
    T4_12 = σ11*σ12 + σ12*σ22 + σ13*σ23
    T4_13 = σ11*σ13 + σ12*σ23 + σ13*σ33
    T4_23 = σ12*σ13 + σ22*σ23 + σ23*σ33

    # T5 = γd⋅γd
    T5_11 = γd11^2 + γd12^2 + γd13^2
    T5_22 = γd12^2 + γd22^2 + γd23^2
    T5_33 = γd13^2 + γd23^2 + γd33^2
    T5_12 = γd11*γd12 + γd12*γd22 + γd13*γd23
    T5_13 = γd11*γd13 + γd12*γd23 + γd13*γd33
    T5_23 = γd12*γd13 + γd22*γd23 + γd23*γd33

    # T6 = σ⋅γd + γd⋅σ
    T6_11 = 2*(σ11*γd11 + σ12*γd12 + σ13*γd13)
    T6_22 = 2*(σ12*γd12 + σ22*γd22 + σ23*γd23)
    T6_33 = 2*(σ13*γd13 + σ23*γd23 + σ33*γd33)
    T6_12 = σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23
    T6_13 = σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33
    T6_23 = σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33

    # T7 = σ⋅σ⋅γd + γd⋅σ⋅σ
    T7_11 = 2*(T4_11*γd11 + T4_12*γd12 + T4_13*γd13)
    T7_22 = 2*(T4_12*γd12 + T4_22*γd22 + T4_23*γd23)
    T7_33 = 2*(T4_13*γd13 + T4_23*γd23 + T4_33*γd33)
    T7_12 = T4_11*γd12 + T4_12*γd22 + T4_13*γd23 + γd11*T4_12 + γd12*T4_22 + γd13*T4_23
    T7_13 = T4_11*γd13 + T4_12*γd23 + T4_13*γd33 + γd11*T4_13 + γd12*T4_23 + γd13*T4_33
    T7_23 = T4_12*γd13 + T4_22*γd23 + T4_23*γd33 + γd12*T4_13 + γd22*T4_23 + γd23*T4_33

    # T8 = σ⋅γd⋅γd + γd⋅γd⋅σ
    T8_11 = 2*(σ11*T5_11 + σ12*T5_12 + σ13*T5_13)
    T8_22 = 2*(σ12*T5_12 + σ22*T5_22 + σ23*T5_23)
    T8_33 = 2*(σ13*T5_13 + σ23*T5_23 + σ33*T5_33)
    T8_12 = σ11*T5_12 + σ12*T5_22 + σ13*T5_23 + T5_11*σ12 + T5_12*σ22 + T5_13*σ23
    T8_13 = σ11*T5_13 + σ12*T5_23 + σ13*T5_33 + T5_11*σ13 + T5_12*σ23 + T5_13*σ33
    T8_23 = σ12*T5_13 + σ22*T5_23 + σ23*T5_33 + T5_12*σ13 + T5_22*σ23 + T5_23*σ33

    # T9 = σ⋅σ⋅γd⋅γd + γd⋅γd⋅σ⋅σ
    T9_11 = 2*(T4_11*T5_11 + T4_12*T5_12 + T4_13*T5_13)
    T9_22 = 2*(T4_12*T5_12 + T4_22*T5_22 + T4_23*T5_23)
    T9_33 = 2*(T4_13*T5_13 + T4_23*T5_23 + T4_33*T5_33)
    T9_12 = T4_11*T5_12 + T4_12*T5_22 + T4_13*T5_23 + T5_11*T4_12 + T5_12*T4_22 + T5_13*T4_23
    T9_13 = T4_11*T5_13 + T4_12*T5_23 + T4_13*T5_33 + T5_11*T4_13 + T5_12*T4_23 + T5_13*T4_33
    T9_23 = T4_12*T5_13 + T4_22*T5_23 + T4_23*T5_33 + T5_12*T4_13 + T5_22*T4_23 + T5_23*T4_33

    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE

    # Compute the integrity basis from scalar invariants
    # λ1 = tr(σ)
    λ1 = σ11 + σ22 + σ33

    # λ2 = tr(σ^2)
    λ2 = T4_11 + T4_22 + T4_33

    # λ3 = tr(γd^2)
    λ3 = T5_11 + T5_22 + T5_33

    # λ4 = tr(σ^3)
    λ4 = σ11*T4_11 + σ22*T4_22 + σ33*T4_33 + 2*(σ12*T4_12 + σ13*T4_13 + σ23*T4_23)

    # λ5 = tr(γd^3)
    λ5 = γd11*T5_11 + γd22*T5_22 + γd33*T5_33 + 2*(γd12*T5_12 + γd13*T5_13 + γd23*T5_23)

    # λ6 = tr(σ^2⋅γd^2)
    λ6 = T4_11*T5_11 + T4_22*T5_22 + T4_33*T5_33 + 2*(T4_12*T5_12 + T4_13*T5_13 + T4_23*T5_23)

    # λ7 = tr(σ^2⋅γd)
    λ7 = (T7_11 + T7_22 + T7_33)/2

    # λ8 = tr(σ⋅γd^2)
    λ8 = (T8_11 + T8_22 + T8_33)/2

    # λ9 = tr(σ⋅γd)
    λ9 = (T6_11 + T6_22 + T6_33)/2

    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE

    # Run the integrity basis through a neural network
    model_inputs = [λ1;λ2;λ3;λ4;λ5;λ6;λ7;λ8;λ9]
    g1,g2,g3,g4,g5,g6,g7,g8,g9 = model_univ(model_inputs, model_weights)  # model_univ not found
    #return 0., 0., 0., 0., 0., 0.   # Generates warning. 
    
    # Tensor combining layer
    F11 = g1 + g2*σ11 + g3*γd11 + g4*T4_11 + g5*T5_11 + g6*T6_11 + g7*T7_11 + g8*T8_11 + g9*T9_11
    F22 = g1 + g2*σ22 + g3*γd22 + g4*T4_22 + g5*T5_22 + g6*T6_22 + g7*T7_22 + g8*T8_22 + g9*T9_22
    F33 = g1 + g2*σ33 + g3*γd33 + g4*T4_33 + g5*T5_33 + g6*T6_33 + g7*T7_33 + g8*T8_33 + g9*T9_33
    F12 = g2*σ12 + g3*γd12 + g4*T4_12 + g5*T5_12 + g6*T6_12 + g7*T7_12 + g8*T8_12 + g9*T9_12
    F13 = g2*σ13 + g3*γd13 + g4*T4_13 + g5*T5_13 + g6*T6_13 + g7*T7_13 + g8*T8_13 + g9*T9_13
    F23 = g2*σ23 + g3*γd23 + g4*T4_23 + g5*T5_23 + g6*T6_23 + g7*T7_23 + g8*T8_23 + g9*T9_23

    #return 0., 0., 0., 0., 0., 0.   # REMOVE ONCE DEBUGGED GE
    return F11,F22,F33,F12,F13,F23  # REINSTATE WHEN DEBUGGED
end

function dudt_univ!(du, u, p, t, gradv, dct)
    # Destructure the parameters
    # println("*** INSIDE dudt_univ! ***")
    # println("typeof(dct): ", typeof(dct))
    # println("dct: ", dct)
    # println("dct keys: ", keys(dct) |> collect)
	n_weights = dct[:n_weights]
    model_weights = p[1:n_weights]
    η0 = p[end - 1]
    τ = p[end]

    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u

    # Specify the velocity gradient tensor
    v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)
    γd22 = 2*v22(t)
    γd33 = 2*v33(t) 
    γd12 = v12(t) + v21(t)
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)

    # Run stress/strain through a TBNN
    γd = [γd11,γd22,γd33,γd12,γd13,γd23]
	F11,F22,F33,F12,F13,F23 = tbnn(u,γd,model_weights, dct[:model_univ])

    # The model differential equations
    dσ11 = η0*γd11/τ - σ11/τ + 2*v11(t)*σ11 + v21(t)*σ12 + v31(t)*σ13 + σ12*v21(t) + σ13*v31(t) - F11/τ
    dσ22 = η0*γd22/τ - σ22/τ + 2*v22(t)*σ22 + v12(t)*σ12 + v32(t)*σ23 + σ12*v12(t) + σ23*v32(t) - F22/τ
    dσ33 = η0*γd33/τ - σ33/τ + 2*v33(t)*σ33 + v13(t)*σ13 + v23(t)*σ23 + σ13*v13(t) + σ23*v23(t) - F33/τ
    dσ12 = η0*γd12/τ - σ12/τ + v11(t)*σ12 + v21(t)*σ22 + v31(t)*σ23 + σ11*v12(t) + σ12*v22(t) + σ13*v32(t) - F12/τ
    dσ13 = η0*γd13/τ - σ13/τ + v11(t)*σ13 + v21(t)*σ23 + v31(t)*σ33 + σ11*v13(t) + σ12*v23(t) + σ13*v33(t) - F13/τ
    dσ23 = η0*γd23/τ - σ23/τ + v12(t)*σ13 + v22(t)*σ23 + v32(t)*σ33 + σ12*v13(t) + σ22*v23(t) + σ23*v33(t) - F23/τ

    # Update in place
    du[1] = dσ11
    du[2] = dσ22
    du[3] = dσ33
    du[4] = dσ12
    du[5] = dσ13
    du[6] = dσ23

	#=
    # Update in place
    du[1] = 0.
    du[2] = 0.
    du[3] = 0.
    du[4] = 0.
    du[5] = 0.
    du[6] = 0.
    return  # REMOVE THIS AND ABVOE du[i] = 0 ONCE DEBUGGED
	=#
end

function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, dct)
    # Define the (default) ODEProblem
	dudt_protocol!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[1], dct)
    prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

    # Remake the problem for different protocols
    function prob_func(prob, i, repeat, dct)
		dudt_remade!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[i], dct)
        remake(prob, f=dudt_remade!, tspan=tspans[i])
    end

    #=
    # https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
    # Automatically switch from Tsit5 to Rodas5 when stiffness is detected
    tsidas_alg = AutoTsit5(Rodas5())
    sol = solve(prob, tsitdas_alg)
    =#

	# EnsembleProblem: https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/
    prob_func1(prob, i, repeat) = prob_func(prob, i, repeat, dct)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func1)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, saveat=dct[:saveat])  # original
	# sim is returned by default
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories, dct)
    loss = 0
	println("loss_univ: trajectories: $trajectories")
    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, dct)
    for k = range(1,trajectories,step=1)
        σ12_pred = results[k][4,:]
        σ12_data = σ12_all[k]
        loss += sum(abs2, σ12_pred - σ12_data)
    end
    loss += 0.01*norm(θ,1)   # L1 norm for sparsification
    return loss
end


function plot_data!(plots, targetk, target_titlek, sol_ude_pre, sol_ude_post, sol_giesekus)
    # For each protocol (target), we compute σ12, N1, and N2 from the UDE and Giesekus solutions

    # Prediction with initial NN weights
    σ12_ude_pre  = sol_ude_pre[4,:] # σ12
    N1_ude_pre   = sol_ude_pre[1,:] - sol_ude_pre[2,:]  # N1 = σ11 - σ22
    N2_ude_pre   = sol_ude_pre[2,:] - sol_ude_pre[3,:]  # N2 = σ22 - σ33

    # Prediction with trained NN weights
    σ12_ude_post = sol_ude_post[4,:]    # σ12
    N1_ude_post  = sol_ude_post[1,:] - sol_ude_post[2,:] # N1 = σ11 - σ22
    N2_ude_post  = sol_ude_post[2,:] - sol_ude_post[3,:] # N2 = σ22 - σ33

    #local σ12_data = sol_giesekus[4,:] # ORIG
    σ12_data = sol_giesekus[4,:]   # σ12
    N1_data  = sol_giesekus[1,:] - sol_giesekus[2,:]  # N1 = σ11 - σ22
    N2_data  = sol_giesekus[2,:] - sol_giesekus[3,:]  # N2 = σ22 - σ33

    # Boolean to check whether to stop the OptimizationFunction
    halt = false
    if maximum(σ12_ude_post) > 10 || maximum(N1_ude_post) > 10 || maximum(N2_ude_post) > 10
        halt = true
        println("halt: true")
    end

    # Plot the data
	plot(xlabel="Time", xlims=(0, dct[:T]), titlefontsize=10, legendfontsize=6, legend=:topright)
    #@show targetk
    # When the blue line is not there, it is because the red and blue lines are superimposed. This means that the converged
    # state is near the initial state. This might happen where you run only a few epochs of the neural network. 

    if targetk == "σ12"
        plot!(sol_ude_post.t,σ12_ude_post, c=:red, lw=1.5, label="UDE-post") 
        plot!(sol_ude_pre.t,σ12_ude_pre, c=:blue, ls=:dash, lw=3, title="σ12, v21=$target_titlek", label="UDE-pre") 
        plot_σ12 = scatter!(sol_giesekus.t[1:2:end],σ12_data[1:2:end], c=:black, m=:o, ms=2, label="Giesekus") 
        push!(plots, plot_σ12)
    elseif targetk == "N1"
        plot!(sol_ude_post.t,N1_ude_post,c=:red, lw=1.5, label="UDE-post") 
        plot!(sol_ude_pre.t,N1_ude_pre, c=:blue, ls=:dash, lw=3, title="N1, v21=$target_titlek", label="UDE-pre")  
        plot_N1 = scatter!(sol_giesekus.t[1:2:end],N1_data[1:2:end], m=:o, ms=2, c=:black, label="Giesekus")
        push!(plots, plot_N1)
    elseif targetk == "N2"
        plot!(sol_ude_post.t,N2_ude_post,c=:red, lw=1.5, label="UDE-post") 
        plot!(sol_ude_pre.t,N2_ude_pre,c=:blue, lw=3, title="N2, v21=$target_titlek", ls=:dash, label="UDE-pre")
        plot_N2 = scatter!(sol_giesekus.t[1:2:end], N2_data[1:2:end], m=:o, ms=2, c=:black, label="Giesekus") 
        push!(plots, plot_N2)
    elseif targetk == "ηE"
        plot!(sol_ude_post.t,-N2_ude_post-N1_ude_post, lw=1.5, c=:red, label="UDE-post") 
        plot!(sol_ude_pre.t,-N2_ude_pre-N1_ude_pre, lw=3, c=:blue, title="ηE=-N1-N2, v21=$target_titlek", ls=:dash, label="UDE-pre") 
        plot_N2N1 = scatter!(sol_giesekus.t,-N2_data-N1_data, c=:black, m=:o, ms=2, label="Giesekus") 
        push!(plots, plot_N2N1)
    end  
    return plots, halt
end

