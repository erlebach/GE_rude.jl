# savedir = "/Users/alex/Documents/Important/College/FSU Misc/Masters Project/Project Code/Time Invariance/" # Make sure it has "/" at the end!
# cd(savedir)

# Author: G. Erlebacher
# date: 2023-05-04
# Optimized Gisekus, debugging UODE
# Transform rude_giesekus_optimized.jl to a scalar equation. 
# New file name: rude_giesekus_optimized_reduced_scalar.jl
# The objective is not accuracy, but running without errors. 

# Changes: 
# Remove BSON, load/save

# Use SizedArray everywhere is Zygote is used
# Use similar() to coerce an array type

#using Debugger
using RelevanceStacktrace
using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays

const max_nb_iter::Int32 = 2  # was 200
const max_nb_protocols::Int32 = 1

# Iniitial conditions and time span
tspan = (0.0, 12.0)
tsave = range(tspan[1],tspan[2],length=50)
σ0 = SizedMatrix{3,3}([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])

# Build the protocols for different 'experiments'
v21_1(t) = 1*cos(t)
gradv_1 = SizedMatrix{3,3}([0., 0., 0., v21_1, 0., 0., 0., 0., 0.])
protocols = [gradv_1]
tspans = [tspan]
tsaves = [tsave]

# Set up the Giesekus model
function dudt_giesekus_opt!(du, σ, p, t, gradv)
    # gradv is a
    # σ: 3x3 tensor
    # Destructure the parameters
    η0, τ, α = p
    ∇v = SizedMatrix{3,3}([0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.])
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
σ0 = SizedMatrix{3,3}([1. 0. 0. ; 0. 2. 0. ; 3. 0. 0.])
σ12_all = Any[]
t_all = Any[]
for k = range(1, max_nb_protocols)
    # protocols[k][4] is a function
    dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
    prob_giesekus = ODEProblem(dudt!, σ0, tspans[k], p_giesekus)
    solve_giesekus = solve(prob_giesekus, Tsit5(), saveat=0.2)
    σ12_data = solve_giesekus[4,:]
    push!(t_all, solve_giesekus.t)
    push!(σ12_all, σ12_data)
end

println("size(σ12_all): ", size(σ12_all))

## Set up the UODE
function dudt_univ_opt!(du, σ, p, t, gradv, model_univ, model_weights)
    # the parameters are [NN parameters, ODE parameters)
    η0, τ = @view p[end-1 : end]
    # SizeMatrix wraps an array of known size to increase efficiency. Not static.
    ∇v = SizedMatrix{3,3}([0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.])  # ERROR with gradv ???
    D = 0.5 .* (∇v .+ transpose(∇v))
    T1 = (η0/τ) .* D     #### ERROR
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)
    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    # Change tbnn to read D
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
	du .= (-σ ./ τ) .+ T1 .+ T2  .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    # Compute elements of the tensor basis
    #I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]  # Should work.  Does not work when multiplied
                                           # by a type that is ReverseDifferentiation
    I  = SizedMatrix{3,3}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])

    # Compute the integrity basis from scalar invariants. Traces. 
    # λ = zeros(2) # must be mutale
    # Important to use similar() to ensure the same type of σ, which can change
    λ = similar(σ, (2,)) # must be mutable and the same type of σ
    # println("σ: ", σ)
    # println("tr(σ): ", tr(σ))
    λ[1] = tr(σ)
    λ[2] = 3
    # println("input to NN, λ: ", λ)
    #g = model_univ(λ, model_weights)    ### Error
    g = re(model_weights)(λ)
    println("==> g: ", g) # should be a Vector
    F = g[1] .* I .+ g[2] .* σ
    # println("==> F: ", F)
    return F
end

# ====================
# from inside loss_univ()
#results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, model_univ,  model_weights)
function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, model_univ,  model_weights)
	dudt_protocol!(du,u,p,t) = dudt_univ_opt!(du, u, p, t, protocols[1][4], model_univ, model_weights)
	prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

	# Remake the problem for different protocols
	function prob_func(prob, i, repeat)
		dudt_remade!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[i][4], model_univ, model_weights)  ##  FIXED: added (du in first arg on RHS)
		remake(prob, f=dudt_remade!, tspan=tspans[i])  ### ERROR
	end

	ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	# sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, 
		    # sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)  # Crashes <<<<<<
	sim = solve(ensemble_prob, Rodas4(), ensemble, trajectories=trajectories, 
		    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)  # Crashes <<<<<<
end

#  [θ; p_system] = single column vector
# 	loss_fn(θ) = loss_univ([θ; p_system], protocols[klist], tspans[klist], σ0, σ12_all, klist, model_univ, model_weights) 
function loss_univ(θ, protocols, tspans, σ0, σ12_all, trajectories,  model_univ,  model_weights)
    # protocols: list
    # tspans: list
    # trajectories: integer
	loss = 0
	results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, trajectories, model_univ,  model_weights)
	for k = range(1, trajectories, step=1)
		σ12_pred = results[k][4,:]
		σ12_data = σ12_all[k]   # ERROR on 2023-05-04_19:24
		loss += sum(abs2, σ12_pred - σ12_data)
	end
	loss += 0.01*norm(θ, 1)
	return loss
end

model_univ = Flux.Chain(Flux.Dense(2, 8, tanh),
                       Flux.Dense(8, 2))
p_model, re = Flux.destructure(model_univ)

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

model_weights = p_model 

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
### Do not use rodas4 for now, at least not with SizedArrays. 

# ERROR: ArgumentError: Converting an instance of ReverseDiff.TrackedReal{Float64, Float64, Nothing} to Float64 is not defined. Please use `ReverseDiff.value` instead.