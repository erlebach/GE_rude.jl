# Tutorial
# https://docs.sciml.ai/DiffEqDocs/dev/features/ensemble/

using DifferentialEquations
using OptimizationOptimisers
using DiffEqFlux  # for ReverseDiffVJP

# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u, p, t) -> 1.01u, 0.5, (0.0, 1.0))

function prob_func(prob, i, repeat)
	remake(prob, u0=rand() * prob.u0)
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=1, saveat=0.2)

# sim[i].prob.u0: initial condition of problem i
println(sim[1].prob)
println(sim[1].prob.u0)
println(sim[1] |> size)
