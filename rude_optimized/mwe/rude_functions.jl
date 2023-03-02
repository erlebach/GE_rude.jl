
function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, dct)
    # Define the (default) ODEProblem
	#dudt_protocol!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[1], dct)
	println("enter ensemble_solve")
	
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
			print("inside prob_func, i= ", i)
			dudt_remade(u, p, t) = dudt_univ_opt(u, p ,t, protocols[i][4], model_univ, n_weights, model_weights) # >>> ERROR
        remake(prob, f=dudt_remade, tspan=tspans[i])
    end


	# EnsembleProblem: https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/
    prob_func1(prob, i, repeat) = prob_func(prob, i, repeat, model_univ, n_weights, model_weights)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func1)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, saveat=dct[:saveat])  # original
	# sim is returned by default
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories, dct)
    loss = 0
	println("===> loss_univ: trajectories: $trajectories")
    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, dct)
    for k = range(1,trajectories,step=1)
		xxx = results[k][2,1,:]  # 2nd row, 1st column: σ12 = σ21
		println("after xxx")
		σ12_pred = results[k][2,1,:]  # 2nd row, 1st column: σ12 = σ21
        σ12_data = σ12_all[k]
        loss += sum(abs2, σ12_pred )  # remove Giesekus solution for testing
        #loss += sum(abs2, σ12_pred - σ12_data)
    end
	println("before loss +=")
    loss += 0.01*norm(θ,1)   # L1 norm for sparsification
	println("after loss +=, loss: ")
	println("   loss: ", loss)
    return loss
end


