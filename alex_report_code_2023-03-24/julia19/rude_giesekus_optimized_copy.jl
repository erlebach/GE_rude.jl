# savedir = "/Users/alex/Documents/Important/College/FSU Misc/Masters Project/Project Code/Time Invariance/" # Make sure it has "/" at the end!
# cd(savedir)

# Author: G. Erlebacher
# date: 2023-05-04
# Optimized Gisekus, debugging UODE
# Transform rude_giesekus_optimized.jl to a scalar equation. 
# New file name: rude_giesekus_optimized_reduced_scalar.jl

#using Debugger
# Error Init(nothing...) not found when using RelevanceStacktrace
# Exit Julia and rerun
# using RelevanceStacktrace
# Error occured without RelevanceStacktrace module
using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays
using BSON: @save, @load

VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 1
const max_nb_iter::Int32 = 2
start_at = 1 # Train from scratch

function write_files(files)
    # Open and close a file for writing. This should empty any 
    # existing file
    for fn in files
        file = open(fn, "w")
        close(file)
    end
end

write_files(["Giesekus_lambda.txt", "Giesekus_g.txt"])

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


function dudt_univ_opt!(dσ, σ, p, t, gradv, model_univ, model_weights)
    # the parameters are [NN parameters, ODE parameters)
    η0, τ = @view p[end-1 : end]

    # Governing equations are for components of the stress tensor
    #σ = @SMatrix [u[1] u[4] 0.; u[4] 0. 0.; 0. 0. u[3]]
    # Store in 3x3 static array for increased efficiency
    # Only σ11, σ12, σ21, σ22, and σ33 are non-zero
    # σ = SA[u[1] u[4] 0.; u[4] u[2] 0.; 0. 0. u[3]]

    # Rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    #∇v = @SMatrix [0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]
    # We could get faster execution by preallocating arrays? But then should 
    # not be static. 
    #∇v = SA[0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.]  # ERROR with gradv ???
    ∇v = SizedMatrix{3,3}([0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.])  
    # ∇v = SA[0. 0. 0. ; 0. 0. 0. ; 0. 0. 0.]
    D = 0.5 .* (∇v .+ transpose(∇v))
    T1 = (η0/τ) .* D 
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    # Change tbnn to read D
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
	#du = -σ ./ τ + T1 + T2  - F ./ τ   # 9 equations (static matrix) (old)
	dσ .= (-σ ./ τ) .+ T1 .+ T2  .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    λ = similar(σ, (9,))

    # Compute elements of the tensor basis
    #I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
    I  = SizedMatrix{3,3}([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])
    T4 = σ * σ
    T5 = D * D
    T6 = σ * D + D * σ
    T7 = T4 * D + D * T4
    T8 = σ * T5 + T5 * σ
    T9 = T4 * T5 + T5 * T4

    # Compute the integrity basis from scalar invariants. Traces. 
    λ[1] = tr(σ)
    λ[2] = tr(T4)
    λ[3] = tr(T5)
    λ[4] = tr(σ*σ*σ)
    λ[5] = tr(D*D*D)
    λ[6] = tr(T6)
    λ[7] = tr(T7)
    λ[8] = tr(T8)
    λ[9] = tr(T9)

    g = re(model_weights)(λ)
    # g = model_univ(λ, model_weights)    ### Error

    # Tensor combining layer
    F = g[1] .* I    +   g[2] .* σ    +   g[3] .* D    +   g[4] .* T4   +   g[5] .* T5 + 
        g[6] .* T6   +   g[7] .* T7   +   g[8] .* T8   +   g[9] .* T9
end

# ====================
function ensemble_solve(θ,ensemble,protocols,tspans,σ0,trajectories,model_univ, model_weights)
#function ensemble_solve(θ,ensemble,protocols,tspans,σ0,trajectories)
	# Define the (default) ODEProblem
	#dudt_protocol!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[1])
    println("protocols: ", protocols)
    println("protocols[1]: ", protocols[1])
    println("typeof(protocols): ", typeof(protocols))
    println("typeof(protocols[1]): ", typeof(protocols[1]))
    # Specialized to only a single function: the v21 protocol that is a function of time: 4th component
	dudt_protocol!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[1][4], model_univ, model_weights)
	prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

	# Remake the problem for different protocols
	function prob_func(prob, i, repeat)
		#dudt_remade!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[i])
        # Specialized to only a single function: the v21 protocol that is a function of time: 4th component
        println("line 127, i: ", i)
        println("line 127, typeof(protocols): ", typeof(protocols))
        println("line 127, protocols[i]: ", protocols[i])
        println("line 127, typeof(protocols[i][4]): ", typeof(protocols[i][4]))
        println("line 127, typeof(protocols[i][4](3.)): ", typeof(protocols[i][4](3.)))
        # protocols[i][4] is a Function
		dudt_remade!(du,u,p,t) = dudt_univ_opt!(du,u,p,t,protocols[i][4], model_univ, model_weights)
		remake(prob, f=dudt_remade!, tspan=tspans[i])
        println("135, after remake")
	end

    println("trajectories: ", trajectories)  # 1
	ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	println("typeof(ensemble_prob): ", typeof(ensemble_prob))
    # println("ensemble_prob.body: ", ensemble_prob.body)  
    # println("ensemble_prob.var: ", ensemble_prob.var)
    println("ensemble: ", ensemble)  
    println("ensemble_prob: ", ensemble_prob)  # Not null
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, 
		    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2);  ### ERROR
    println("after sim = solve")
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories, model_univ, model_weights)
	loss = 0
	#results = ensemble_solve(θ,EnsembleThreads(),protocols,tspans,σ0,trajectories)
	results = ensemble_solve(θ,EnsembleThreads(),protocols,tspans,σ0,trajectories,model_univ, model_weights)
	for k = range(1,trajectories,step=1)
		σ12_pred = results[k][4,:]
		σ12_data = σ12_all[k]
		loss += sum(abs2, σ12_pred - σ12_data)
	end
	loss += 0.01*norm(θ, 1)
	return loss
end
# ====================

# Define the simple shear deformation protocol
v11(t) = 0
v12(t) = 0
v13(t) = 0
v22(t) = 0
v23(t) = 0
v31(t) = 0
v32(t) = 0
v33(t) = 0

# Iniitial conditions and time span
tspan = (0.0f0, 12f0)
tsave = range(tspan[1],tspan[2],length=50)
σ0 = SizedMatrix{3,3}([0f0 0f0 0f0; 0f0 0f0 0f0; 0f0 0f0 0f0])

# Build the protocols for different 'experiments'
ω = 1f0
v21_1(t) = 1*cos(ω*t)
v21_2(t) = 2*cos(ω*t)
v21_3(t) = 1*cos(ω*t/2)
v21_4(t) = 2*cos(ω*t/2)
v21_5(t) = 1*cos(2*ω*t)
v21_6(t) = 2*cos(2*ω*t)
v21_7(t) = 1*cos(ω*t/3)
v21_8(t) = 2*cos(ω*t/3)
gradv_1 = [v11,v12,v13,v21_1,v22,v23,v31,v32,v33]
gradv_2 = [v11,v12,v13,v21_2,v22,v23,v31,v32,v33]
gradv_3 = [v11,v12,v13,v21_3,v22,v23,v31,v32,v33]
gradv_4 = [v11,v12,v13,v21_4,v22,v23,v31,v32,v33]
gradv_5 = [v11,v12,v13,v21_5,v22,v23,v31,v32,v33]
gradv_6 = [v11,v12,v13,v21_6,v22,v23,v31,v32,v33]
gradv_7 = [v11,v12,v13,v21_7,v22,v23,v31,v32,v33]
gradv_8 = [v11,v12,v13,v21_8,v22,v23,v31,v32,v33]
protocols = [gradv_1, gradv_2, gradv_3, gradv_4, gradv_5, gradv_6, gradv_7, gradv_8]
tspans = [tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan]
tsaves = [tsave, tsave, tsave, tsave, tsave, tsave, tsave, tsave]
#println("AFTER tsaves, typeof(protocols[1]): ", typeof(protocols[1]))
#println("AFTER tsaves, typeof(protocols[1][4]): ", typeof(protocols[1][4]))
#println("typeof v21_1: $(typeof(v21_1))")

# Solve for the Giesekus model
η0 = 1
τ = 1
α = 0.8
p_giesekus = [η0,τ,α]
σ12_all = Any[]
t_all = Any[]
for k = range(1,max_nb_protocols)
    println("**************************************")
	#dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
	dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
    tspans[1] = (0f0, 12f0)
    #What does {true, SciMLBase.FullSpecialize} do?
	prob_giesekus = ODEProblem{true, SciMLBase.FullSpecialize}(dudt!, σ0, tspans[k], p_giesekus)
	#prob_giesekus = ODEProblem(dudt!, σ0, tspans[k], p_giesekus)
	solve_giesekus = solve(prob_giesekus, Rodas4(), saveat=0.2) 
	σ12_data = solve_giesekus[4,:]
    print("==> ", σ12_data)
	push!(t_all, solve_giesekus.t)
	push!(σ12_all, σ12_data)
end

# NN model for the nonlinear function F(σ,γ̇)
model_univ = Flux.Chain(Flux.Dense(9, 8, tanh),
                       #Flux.Dense(8, 8, tanh),
                       Flux.Dense(8, 9))
p_model, re = Flux.destructure(model_univ)

# The protocol at which we'll start continuation training
# (choose start_at > length(protocols) to skip training)
# start_at = 9 # Don't train
#protocols = protocols[1:1] # ONLY RUN a single protocol (for speed, for now)

if start_at > 1
	# Load the pre-trained model if not starting from scratch
	@load "tbnn.bson" θi
	p_model = θi
	n_weights = length(θi)
else
	# The model weights are destructured into a vector of parameters
	n_weights = length(p_model)
	p_model = zeros(n_weights)
end

# Parameters of the linear response (η0,τ)
p_system = Float32[1, 1]

θ0 = zeros(size(p_model))
θi = p_model

# Callback function to print the iteration number and loss
iter = 0
callback = function (θ, l, protocols, tspans, σ0, σ12_all, trajectories)
  global iter
  iter += 1
  println(l)
  println(iter)
  return false
end

model_weights = p_model # + [1., 1., 1.]


# Continutation training loop
adtype = Optimization.AutoZygote()
# for k = range(start_at,length(protocols),step=1)
for k = range(1,max_nb_protocols)
    println("==> k= $k")
	#loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k)
	loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k, model_univ, model_weights) # <<<< ERROR
	#loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k, model_univ, n_weights_, model_weights) # <<<< ERROR
	cb_fun(θ, l) = callback(θ, l, protocols[1:k], tspans[1:k], σ0, σ12_all, k)
    # println("after callback")
	optf = Optimization.OptimizationFunction((x,p) -> loss_fn(x), adtype)
    # println("after optf")
	optprob = Optimization.OptimizationProblem(optf, θi)
    # println("after Optimization")
	result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_fun, maxiters=max_nb_iter)  ### ERROR
	# println("after result_univ")
	# global θi = result_univ.u
	# @save "tbnn.bson" θi
    # @save "tbnn_protocol$k.bson" θi
end   ### model_weights not defined

# Build full parameter vectors for model testing
θ0 = [θ0; p_system]
θi = [θi; p_system]

#=
### ERROR
line 127, typeof(protocols[i][4](3.)): Float64
135, after remake
ERROR: MethodError: no method matching init(::Nothing, ::Tsit5{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(OrdinaryDiffEq.trivial_limiter!), Static.False}; sensealg::InterpolatingAdjoint{0, true, Val{:central}, ReverseDiffVJP{true}}, saveat::Float64)

Closest candidates are:
  init(::OptimizationProblem, ::Any, ::Any...; kwargs...)
   @ SciMLBase ~/.julia/packages/SciMLBase/VdcHg/src/solve.jl:146
  init(::PDEProblem, ::SciMLBase.AbstractDEAlgorithm, ::Any...; kwargs...)
   @ DiffEqBase ~/.julia/packages/DiffEqBase/ZXMKG/src/solve.jl:1050
  init(::SciMLBase.AbstractJumpProblem, ::Any...; kwargs...)
   @ DiffEqBase ~/.julia/packages/DiffEqBase/ZXMKG/src/solve.jl:443
  ...
  =#