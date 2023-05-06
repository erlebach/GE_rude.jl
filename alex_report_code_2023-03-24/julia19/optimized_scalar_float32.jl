

# savedir = "/Users/alex/Documents/Important/College/FSU Misc/Masters Project/Project Code/Time Invariance/" # Make sure it has "/" at the end!
# cd(savedir)

# Author: G. Erlebacher
# date: 2023-05-05
# Optimized Gisekus, debugging UODE
# Transform rude_giesekus_optimized.jl to a scalar equation. 
# New file name: rude_giesekus_optimized_reduced_scalar.jl
# The objective is not accuracy, but running without errors. 
# Convert rude_giesekus_optimized_reduced_scalar to work fully in Float32
## Is there a way to convert from Float32 to Float64 easily as is possible in C++ using typedef? 

# Changes: 
# Remove BSON, load/save

#using Debugger
using RelevanceStacktrace
using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays

const max_nb_iter::Int32 = 100
const max_nb_protocols::Int32 = 1

# Set up the Giesekus model
function dudt_giesekus_opt!(du, σ, p, t, gradv)
    # gradv is a
    # σ: 3x3 tensor
    # Destructure the parameters
    η0, τ, α = p
    ∇v = SA[0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]
    D = 0.5 .* (∇v .+ transpose(∇v))
    T1 = (η0/τ) * D
    T2 = (transpose(∇v) * σ) + (σ * ∇v)
    coef = α / (τ * η0)
    F = coef * (σ * σ)
    du .= -σ / τ .+ T1 .+ T2  .- F  # 9 equations (static matrix)
end

# Solve for the Giesekus model
# Solution is required in order to compute the loss function for the UODE
η0 = 1
τ = 1
α = 0.8
p_giesekus = [η0,τ,α]
σ12_all = Any[]
t_all = Any[]
# for k = range(1,length(protocols),step=1)
for k = range(1, max_nb_protocols)
    # protocols[k][4] is a function
    dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
    prob_giesekus = ODEProblem(dudt!, σ0, tspans[k], p_giesekus)
    solve_giesekus = solve(prob_giesekus,Rodas4(),saveat=0.2)
    σ12_data = solve_giesekus[4,:]
    push!(t_all, solve_giesekus.t)
    push!(σ12_all, σ12_data)
end

println("size(σ12_all): ", size(σ12_all))

## Set up the UODE
function dudt_univ_opt!(du, σ, p, t, gradv, model_univ, model_weights)
    # the parameters are [NN parameters, ODE parameters)
    η0, τ = @view p[end-1 : end]

    # Governing equations are for components of the stress tensor
    #σ = @SMatrix [u[1] u[4] 0.; u[4] 0. 0.; 0. 0. u[3]]
    # Store in 3x3 static array for increased efficiency
    # Only σ11, σ12, σ21, σ22, and σ33 are non-zero
    # σ = SA[u[1] u[4] 0.; u[4] u[2] 0.; 0. 0. u[3]]
    ∇v = SA[0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]  # ERROR with gradv ???
    D = 0.5 .* (∇v .+ transpose(∇v))
    @show η0, τ
    @show D
    #=
    D = ReverseDiff.TrackedReal{Float64, Float32, Nothing}[TrackedReal<66I>(0.0, 0.0, ---, ---) 
       TrackedReal<KLf>(0.42192697525024414, 0.0, tjx, ---) TrackedReal<LeI>(0.0, 0.0, ---, ---); 
       TrackedReal<8QP>(0.42192697525024414, 0.0, tjx, ---) TrackedReal<EfV>(0.0, 0.0, ---, ---) TrackedReal<4WZ>(0.0, 0.0, ---, ---); 
       TrackedReal<i84>(0.0, 0.0, ---, ---) TrackedReal<Czb>(0.0, 0.0, ---, ---) TrackedReal<Bau>(0.0, 0.0, ---, ---)]
    =#
    T1 = (η0/τ) .* D     #### ERROR
    @show T1
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)

    # All of a sudden, 
    # (η0, τ) = (TrackedReal<HYQ>(1.0, 0.0, 6M7, 26, 4yZ), TrackedReal<LSH>(1.0, 0.0, 6M7, 27, 4yZ))

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    # Change tbnn to read D
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
    # @show σ, τ, T1, T2, F
	du .= (-σ ./ τ) .+ T1 .+ T2  .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    # Compute elements of the tensor basis
    I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]

    # Compute the integrity basis from scalar invariants. Traces. 
    λ = zeros(1) # must be mutale
    λ[1] = tr(σ)
    # println("input to NN, λ: ", λ)
    #g = model_univ(λ, model_weights)    ### Error
    g = re(model_weights)(λ)
    # println("==> g: ", g) # should be a Vector
    F = g[1] .* I 
end

# ====================
# from inside loss_univ()
#results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, model_univ,  model_weights)
function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, model_univ,  model_weights)
    # protocols: list of lists
    # protocols: tspans: list of sequences
    # trajectories: sequence (e.g., 1:5)
    # println("==> ensemble_solve, protocols: ", protocols)
	dudt_protocol!(du,u,p,t) = dudt_univ_opt!(du, u, p, t, protocols[1][4], model_univ, model_weights)
	prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)
    # println("prob: ", prob)

	# Remake the problem for different protocols
	function prob_func(prob, i, repeat)
        # println("==> prob_func, i: ", i)
        # println("protocols[i]: ", protocols[i])
        # println("model_univ: ", model_univ)
        # println("model_weights: ", model_weights)
		dudt_remade!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[i][4], model_univ, model_weights)  ##  FIXED: added (du in first arg on RHS)
		remake(prob, f=dudt_remade!, tspan=tspans[i])  ### ERROR
	end

	ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    # # # println("\n==> after ensemble_prob: ", ensemble_prob)
    # # println("\ntrajectories: ", trajectories)
    # println("\nensemble: ", ensemble)  # EnsembleThreads()
    #sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories,
    #sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)))
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, 
		    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)  # Crashes <<<<<<
    # error("ENSEMBLE ERROR 2")  # Line not reached
end

#  [θ; p_system] = single column vector
# 	loss_fn(θ) = loss_univ([θ; p_system], protocols[klist], tspans[klist], σ0, σ12_all, klist, model_univ, model_weights) 
function loss_univ(θ, protocols, tspans, σ0, σ12_all, trajectories,  model_univ,  model_weights)
    # protocols: list
    # tspans: list
    # trajectories: integer
	loss = 0
    println("protocols: ", protocols)
    println("trajectories: ", trajectories)
    println("tspans: ", tspans)
	#results = ensemble_solve(θ,EnsembleThreads(),protocols,tspans,σ0,trajectories)
	results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, model_univ,  model_weights)
    #error("GORDON, loss_univ, after ensemble_solve")  # Line reached at 2023-05-04_19:23
	for k = range(1, trajectories, step=1)
        # print("\n\n==> ")
        @show k 
        # print("\n\n==> ")
        # @show typeof(results)
        # print("\n\n==> ")
        # @show results
        # print("\n\n==> ")
        # @show results[k]
        # print("\n\n==> ")
        println("==> size(σ12_all): ", size(σ12_all))  # Size 0. WHY? 
        # println("==> σ12_all[k]: ", σ12_all[k])
        # println("==> size(results): ", size(results))
        # println("==> size(results[k]): ", size(results[k]))
		σ12_pred = results[k][4,:]
		σ12_data = σ12_all[k]   # ERROR on 2023-05-04_19:24
        println("INSIDE LOSS")
        @show σ12_pred
        @show σ12_data
		loss += sum(abs2, σ12_pred - σ12_data)
        # error("LOSS_UNIV")  # point reached
	end
	loss += 0.01*norm(θ, 1)
	return loss
end
# ====================
# Iniitial conditions and time span
tspan = (0.0f0, 12f0)
tsave = range(tspan[1],tspan[2],length=50)
σ0 = [0f0 0f0 0f0; 0f0 0f0 0f0; 0f0 0f0 0f0]

# Build the protocols for different 'experiments'
v21_1(t) = 1*cos(t)
gradv_1 = [0., 0., 0., v21_1, 0., 0., 0., 0., 0.]
protocols = [gradv_1]
tspans = [tspan]
tsaves = [tsave]

# NN model for the nonlinear function F(σ,γ̇)
model_univ = Flux.Chain(Flux.Dense(1, 8, tanh),
                       Flux.Dense(8, 1))
p_model, re = Flux.destructure(model_univ)

# println("p_model: ", p_model)
# println("re: ", re)

# Parameters of the linear response (η0,τ)
p_system = Float32[1, 1]

θ0 = zeros(size(p_model))
θi = p_model

# Callback function to print the iteration number and loss
iter = 0
callback = function (θ, l)  #, protocols, tspans, σ0, σ12_all, trajectories)
  global iter
  iter += 1
  return false
end

model_weights = p_model # + [1., 1., 1.]

# Continutation training loop
adtype = Optimization.AutoZygote()
    klist = [1,1]
    trajectory = klist[1]
	loss_fn(θ) = loss_univ([θ; p_system], protocols[klist], tspans[klist], σ0, σ12_all, trajectory, model_univ, model_weights) 
	cb_fun(θ, l) = callback(θ, l)
	optf = Optimization.OptimizationFunction((x,p) -> loss_fn(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, θi)
	result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_fun, maxiters=max_nb_iter)  ### ERROR


#### Make sure I am only working with Float32.
### D = ReverseDiff.TrackedReal{Float64, Float32, Nothing}[TrackedReal<66I>(0.0, 0.0, ---, ---) 
### I don't understand, but notice the {Float64, Float32, Nothing}