#using DiffEqFlux, Flux, Optim, DifferentialEquations, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using DiffEqFlux, Flux, Optim, DifferentialEquations# ,LinearAlgebra #, OrdinaryDiffEq
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Zygote
using StaticArrays


function dudt_univ_opt(u, p, t)
	du = @SMatrix [0. 0. 0.; 0. 0. 0.; 0. 0. 0.]  # Static array
end

function single_run()
	# Define the simple shear deformation protocol (all components except v21(t))
    v11(t) = 0
    
    # Initial conditions and time span
	tspan = (0.0f0, 1.0f0)
	σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]  # Generates set_index! error

	p_system = Float32[0., 0.]  # change 2023-02-26_15:40
	θi = [0.] 

	# Loss function closure (first parameter: concatenate all parameters)
	loss_fn(θ) = loss_univ([θ; p_system], tspan, σ0, 1) 
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p)->loss_fn(x),  adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)  
		parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=20)
end

function ensemble_solve(θ, ensemble, tspans, σ0, trajectories)
	θ = [1]
    prob = ODEProblem(dudt_univ_opt, σ0, tspans[1], θ)

    function prob_func(prob, i, repeat)
			dudt_remade(u, p, t) = dudt_univ_opt(u, p ,t) 
        remake(prob, f=dudt_remade, tspan=tspans[i])
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, saveat=0.2)  
end

function loss_univ(θ, tspans, σ0, trajectories)
	println("ENTER loss_univ")
    loss = 0
    results = ensemble_solve(θ, EnsembleThreads(), tspans, σ0, trajectories)
	println("results: ", results |> size)
	println("results[1]: ", results[1] |> size)
	xxx = results[1][2,1,:]  # 2nd row, 1st column: σ12 = σ21
	println("results[1][2,1,:]: ==> ", xxx)
	σ12_pred = results[1][2,1,:]  # 2nd row, 1st column: σ12 = σ21
	println("σ12_pred: ", σ12_pred)
    loss += sum(abs2, σ12_pred )  # Remove this line and error changes
	println("before exiting loss_univ, loss: ", loss)
    return loss
end

single_run()
