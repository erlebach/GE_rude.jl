using DifferentialEquations  # for Tsit5
using OptimizationOptimisers
using DiffEqFlux  # for ReverseDiffVJP


function dudt_univ_opt(u, p, t)
	du = [0.]
end

function single_run()
	tspan = (0.0f0, 1.0f0)
	σ0 = [0.]

	p_system = Float32[0., 0.]  
	θi = [0.] 

	loss_fn(θ) = loss_univ([θ; p_system], tspan, σ0)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x,p)->loss_fn(x),  adtype)
    optprob = Optimization.OptimizationProblem(optf, θi)  
	parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=20)
end

function ensemble_solve(θ, ensemble, tspans, σ0) 
	θ = [1.]
    prob = ODEProblem(dudt_univ_opt, σ0, tspans[1], θ)

    function prob_func(prob, i, repeat)
        remake(prob, f=dudt_univ_opt, tspan=tspans[i])
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=1:1)
end

function loss_univ(θ, tspans, σ0) 
    loss = 0.
	results = ensemble_solve(θ, EnsembleThreads(), tspans, σ0) 
    return loss
end

single_run()
