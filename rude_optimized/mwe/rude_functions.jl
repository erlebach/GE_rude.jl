
function single_run(dct)

	# Define the simple shear deformation protocol (all components except v21(t))
    v11(t) = 0
    
    # Initial conditions and time span
	tspan = (0.0f0, 1.0f0)

	σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]  # Generates set_index! error
	v21_protoc = [ (t) -> 0. for i in 1:8]

	# Linear array of size 9
	protocols = [ [v11,v11,v11,  v21_protoc[1], v11,v11,v11,v11,v11] ]
    nb_protoc = 1
    
	p_system = Float32[0., 0.]  # change 2023-02-26_15:40
    
	θi = [0.] #p_model

	# Loss function closure (first parameter: concatenate all parameters)
	loss_fn(θ) = loss_univ([θ; p_system], protocols[1], tspan, σ0, [0.], 1) #dct)
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p)->loss_fn(x),  adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)  
		parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=20)
    #end
end


function dudt_univ_opt(u, p, t)
	du = @SMatrix [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]  # Static array
end

function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories)
	dudt_protocol(u, p, t) = dudt_univ_opt(u, p ,t)
	θ = [1]
    prob = ODEProblem(dudt_protocol, σ0, tspans[1], θ)

    function prob_func(prob, i, repeat)
			dudt_remade(u, p, t) = dudt_univ_opt(u, p ,t) # >>> ERROR
        remake(prob, f=dudt_remade, tspan=tspans[i])
    end

	# EnsembleProblem: https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/
    prob_func1(prob, i, repeat) = prob_func(prob, i, repeat)
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func1)
	saveat = 0.2
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, saveat=0.2)  # original
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories)
    loss = 0
    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories)
    for k = range(1,trajectories,step=1)
		xxx = results[k][2,1,:]  # 2nd row, 1st column: σ12 = σ21
		σ12_pred = results[k][2,1,:]  # 2nd row, 1st column: σ12 = σ21
        loss += sum(abs2, σ12_pred )  # remove Giesekus solution for testing
    end
    loss += 0.01*norm(θ,1)   # L1 norm for sparsification
    return loss
end


