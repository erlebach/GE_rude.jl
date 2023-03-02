function dudt_univ_opt(u, p, t, gradv, model_univ, n_weights, model_weights)
	du = @SMatrix [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]  # Static array
	#du = zeros(Float64,3,3)  # regular Array (creates another error)
	#ERROR: LoadError: OrdinaryDiffEq.TypeNotConstantError(StaticArraysCore.SMatrix{3, 3, Float64, 9}, Matrix{Float64})
end

function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, dct)
	# Extracting data from dictionaries does type inference, so will only occur once, not each time the element 
	# is accessed with the solver is called at different times steps. 
	model_weights = dct[:model_weights]
	n_weights = dct[:n_weights]
	model_univ = dct[:model_univ]
	# protocols[1][4] is the ∇v21 component of the velocity gradient, externally imposed. 
	dudt_protocol(u, p, t) = dudt_univ_opt(u, p ,t, protocols[1][4], model_univ, n_weights, model_weights)
    prob = ODEProblem(dudt_protocol, σ0, tspans[1], θ)

    # Remake the problem for different protocols
    function prob_func(prob, i, repeat, model_univ, n_weights, model_weights)
			dudt_remade(u, p, t) = dudt_univ_opt(u, p ,t, protocols[i][4], model_univ, n_weights, model_weights) # >>> ERROR
        remake(prob, f=dudt_remade, tspan=tspans[i])
    end

	println("=========================")
	println("++> trajectories: ", trajectories)    # Train the model from scratch
	println("--> trajectories: ", trajectories)    # 1
	println("--> trajectories: ", trajectories)    # 1
	println("--> trajectories: ", trajectories)    # 1
	println("=========================")

	# EnsembleProblem: https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/
    prob_func1(prob, i, repeat) = prob_func(prob, i, repeat, model_univ, n_weights, model_weights)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func1)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, saveat=dct[:saveat])  # original
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories, dct)
    loss = 0
    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, dct)
    for k = range(1,trajectories,step=1)
		xxx = results[k][2,1,:]  # 2nd row, 1st column: σ12 = σ21
		σ12_pred = results[k][2,1,:]  # 2nd row, 1st column: σ12 = σ21
        loss += sum(abs2, σ12_pred )  # remove Giesekus solution for testing
    end
    loss += 0.01*norm(θ,1)   # L1 norm for sparsification
    return loss
end


