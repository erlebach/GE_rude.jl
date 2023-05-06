# using RelevanceStacktrace
using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays

VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 1
const max_nb_iter::Int32 = 5
start_at = 1 # Train from scratch

function dudt_giesekus_opt!(du, σ, p, t, gradv)
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
    ∇v = SizedMatrix{3,3}([0. 0. 0. ; gradv(t) 0. 0. ; 0. 0. 0.])  
    D = 0.5 .* (∇v .+ transpose(∇v))
    T1 = (η0/τ) .* D 
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
	dσ .= (-σ ./ τ) .+ T1 .+ T2  .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    λ = similar(σ, (9,))

    # Compute elements of the tensor basis
    #I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]  # Generates an error. Already reported.
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

    F = g[1] .* I    +   g[2] .* σ    +   g[3] .* D    +   g[4] .* T4   +   g[5] .* T5 + 
        g[6] .* T6   +   g[7] .* T7   +   g[8] .* T8   +   g[9] .* T9
end

# ====================
function ensemble_solve(θ,ensemble,protocols,tspans,σ0,trajectories,model_univ, model_weights)
	dudt_protocol!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[1][4], model_univ, model_weights)
	prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

	# Remake the problem for different protocols
	function prob_func(prob, i, repeat)
		dudt_remade!(du,u,p,t) = dudt_univ_opt!(du,u,p,t,protocols[i][4], model_univ, model_weights)
		remake(prob, f=dudt_remade!, tspan=tspans[i])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, 
		    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2);  ### ERROR
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories, model_univ, model_weights)
	loss = 0
	results = ensemble_solve(θ,EnsembleThreads(),protocols,tspans,σ0,trajectories,model_univ, model_weights)
	for k = range(1,trajectories,step=1)
		σ12_pred = results[k][4,:]
		σ12_data = σ12_all[k]
		loss += sum(abs2, σ12_pred - σ12_data)
	end
	loss += 0.01 * norm(θ, 1)
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

# Solve for the Giesekus model
η0 = 1
τ = 1
α = 0.8
p_giesekus = [η0,τ,α]
σ12_all = Any[]
t_all = Any[]
for k = range(1,max_nb_protocols)
	dudt!(du,u,p,t) = dudt_giesekus_opt!(du,u,p,t,protocols[k][4])
    tspans[1] = (0f0, 12f0)
	prob_giesekus = ODEProblem{true, SciMLBase.FullSpecialize}(dudt!, σ0, tspans[k], p_giesekus)
	solve_giesekus = solve(prob_giesekus, Rodas4(), saveat=0.2) 
	σ12_data = solve_giesekus[4,:]
	push!(t_all, solve_giesekus.t)
	push!(σ12_all, σ12_data)
end

# NN model for the nonlinear function F(σ,γ̇)
model_univ = Flux.Chain(Flux.Dense(9, 8, tanh),
                       #Flux.Dense(8, 8, tanh),
                       Flux.Dense(8, 9))
p_model, re = Flux.destructure(model_univ)

n_weights = length(p_model)
p_model = zeros(n_weights)

# Parameters of the linear response (η0,τ)
p_system = Float32[1, 1]

θ0 = zeros(size(p_model))
θi = p_model

#= =#
# Callback function to print the iteration number and loss
iter = 0
callback = function (θ, l, protocols, tspans, σ0, σ12_all, trajectories)
  global iter
  iter += 1
  println("===> Loss($iter): $(round(l, digits=4)), $l")
  return false
end
#= =#

model_weights = p_model # + [1., 1., 1.]


# Continutation training loop
adtype = Optimization.AutoZygote()
k = 1
iter = 0
for k = range(1,max_nb_protocols)
	loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k, model_univ, model_weights) 
	cb_fun(θ, l) = callback(θ, l, protocols[1:k], tspans[1:k], σ0, σ12_all, k)
	optf = Optimization.OptimizationFunction((x,p) -> loss_fn(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, θi)
	result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_fun, maxiters=max_nb_iter)  
end   


