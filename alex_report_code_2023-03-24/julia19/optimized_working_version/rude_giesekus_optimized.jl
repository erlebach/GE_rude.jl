# using RelevanceStacktrace
using Revise
using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays

# substitute giesekus_impl by PTT_impl to change the base equations
# Implementations of the needed functions. 
# Use includet with Revise
includet("giesekus_impl.jl")

# Set up Parameters
VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 1
const max_nb_iter::Int32 = 5
start_at = 1 # Train from scratch
ω = 1f0

protocols, tspans, tsaves = setup_protocols(ω)
σ0 = SizedMatrix{3,3}([0f0 0f0 0f0; 0f0 0f0 0f0; 0f0 0f0 0f0])

# Parameters for the Giesekus model
η0, τ, α = [1, 1, 0.8]
p_giesekus = [η0, τ, α]
k = 1

# Solve base model with no NN
t_all, σ12_all = solve_giesekus_protocols(protocols, tspans, p_giesekus, σ0, max_nb_protocols);

# Set up Neural Netoork
model_univ = NeuralNetwork(nb_in=9, nb_out=9, layer_size=8, nb_hid_layers=1)
p_model, re = Flux.destructure(model_univ)
n_weights = length(p_model)
p_model = zeros(n_weights)
model_weights = p_model # + [1., 1., 1.]

# Parameters of the linear response (η0,τ) + NN weights
p_system = p_giesekus[1:2]

# Zero weight initialization
θ0 = zeros(size(model_weights))
θi = model_weights

# Callback function to print the iteration number and loss
iter = 0
callback = function (θ, l, protocols, tspans, σ0, σ12_all, trajectories)
  global iter
  iter += 1
  println("===> Loss($iter): $(round(l, digits=4)), $l")
  return false
end

# Solve the UODE
solve_UODE(θi, max_nb_protocols, p_system, tspans, σ0, σ12_all, model_univ, loss_univ, model_weights, max_nb_iter)