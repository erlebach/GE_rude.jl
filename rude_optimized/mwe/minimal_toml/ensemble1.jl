# Tutorial
# https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/
#
# Create a loss function. 

using DifferentialEquations
using OptimizationOptimisers
using DiffEqFlux  # for ReverseDiffVJP

# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0

#function ensemble_solve(θ, ensemble, tspans, σ0) 
function ensemble_solve()
	prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))

	function prob_func(prob, i, repeat)
		remake(prob, u0=rand() * prob.u0)
	end

	ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
	sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=1, saveat=0.2)
end

function loss_univ(θ, tspans, σ0) 
    loss = 0.
	#results = ensemble_solve(θ, EnsembleThreads(), tspans, σ0) 
	results = ensemble_solve()
	print("results[1]: ", results[1])

    #prob = ODEProblem(dudt_eq, σ0, tspans[1], θ)
	#prob_ode = ODEProblem(dudt_eq, σ0, tspans[1])
	#solve(prob_ode, Tsit5(), saveat=0.2)
    return loss
end

θ, tspan, σ0 = [.1], (0., 1.), [0.01]
loss_univ(θ, tspan, σ0)
