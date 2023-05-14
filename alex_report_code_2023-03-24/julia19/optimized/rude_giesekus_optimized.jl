# using RelevanceStacktrace
using Revise
using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using StaticArrays
using StableRNGs
# using Colors

# substitute giesekus_impl by PTT_impl to change the base equations
# Implementations of the needed functions. 
# Use includet with Revise
includet("giesekus_impl.jl")

# Plotting and processing of the solution
includet("myplots.jl")

# ========== START PARAMETER SETUP =====================================
# Set up Parameters
VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 8
const max_nb_iter::Int32 = 100   # 2023-05-10_18:26, why am I not getting 100 iterations per protocol?
start_at = 1 # Train from scratch
tspan = (0., .0001)
tspan = (0., 12.)
ω = 1.0

n_protocol_train = 8
ampl =  SizedVector{n_protocol_train}([1., 2., 1., 2., 1., 2., 1., 2.])
freq =  SizedVector{n_protocol_train}([1., 1., 0.5, 0.5, 2., 2., 1/3., 1/3.])
v21_fct = Any[t -> ampl[i] * cos(freq[i]*ω*t) for i in range(1, n_protocol_train, step=1)]
# ========== END PARAMETER SETUP =====================================

# Set up the protocols
protocols, tspans, tsaves = setup_protocols(n_protocol_train, v21_fct, tspan) 

# σ0 = SizedMatrix{3,3}([0f0 0f0 0f0; 0f0 0f0 0f0; 0f0 0f0 0f0])
σ0 = SizedMatrix{3,3}([0. 0. 0.; 0. 0. 0.; 0. 0. 0.])

# Parameters for the Giesekus model
# Could use a named tuple? (η0=1, τ=1, α=0.8)
η0, τ, α = [1, 1, 0.8]
p_giesekus = [η0, τ, α]

#********************************
# Solve base model with no NN
#********************************
 t_all, σ12_all = solve_giesekus_protocols(protocols, tspans, p_giesekus, σ0, max_nb_protocols);

 #---------------------------------------------------------
#  # Execute the Giesekus code with random I.C. and compare to my new code
# rng =  StableRNG(1234)
# σ0 = rand(rng, 8)
# p = p_giesekus

# Set up Neural Netoork (input layer, hidden layers, output layer)
# I would also like to run in Float32 later once debugged.
model_univ = NeuralNetwork(nb_in=9, nb_out=9, layer_size=8, nb_hid_layers=0)

for p in Flux.params(model_univ)
    fill!(p, 0.00)
end

# p_model becomes a single 1D linear vector. Needed by the UODE optimizer. 
p_model, re = Flux.destructure(model_univ)
n_weights = length(p_model)
# new for debugging and compatibility with non-optimized code
p_model = zeros(n_weights)  # <<<<<<<< NEW
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
  println("===> Loss($iter): $(round(l, digits=4))")
  return false
end

# ==============================================================
if false
# Make the model weights random for debugging
function check_giesekus_opt_NN()
    global re
    rng = StableRNG(1234)
    u0 = rand(rng, 6)
    σ0 = rand(rng, 3, 3)
    σ0[1,1], σ0[2,2], σ0[3,3], σ0[1,2], σ0[1,3], σ0[2,3] = [u0[i] for i in 1:6]
    σ0[2,1] = σ0[1,2]
    σ0[3,1] = σ0[1,3]
    σ0[3,2] = σ0[2,3]

    p_giesekus = [1., 1.]  # Must be a list
    t = 3.
    gradv = [t -> 0.  t -> 0.  t -> 0.; t -> cos(t)  t -> 0.  t -> 0.; t -> 0.  t -> 0.  t -> 0.]
    # println("1. gradv: ", [grad(t) for grad in gradv])
    du = similar(σ0, 3, 3)

    model_univ = NeuralNetwork(; nb_in=9, nb_out=9, layer_size=8, nb_hid_layers=0)
    p_model, re = Flux.destructure(model_univ)
    # p_model = zeros(size(p_model))  # zero weights in a single list
    rng = StableRNG(4321)
    p_model = 0.01 .* (-1. .+ 2 .* rand(rng, size(p_model)[1]))  # zero weights in a single list

    # @show p_model
    p = [p_model; p_giesekus]
    model_weights = p_model
    # println("len p: ", p|>length)
                                             #           <<<  re  >>>>
    dudt_univ_opt!(du, σ0, p, t, gradv, model_univ, model_weights)
    # @show model_weights
    # println("du: ", du)
    # println("du symmetric? ", is_symmetric(du))
    # println("\ncheck_giesekus_opt_NN: ")
    display(du)
    println("==================================================")
    return du
end

check_giesekus_opt_NN()
end
# ==============================================================

#************************************************************************
# Solve the UODE
function reset()
    global rng = StableRNG(1234)
    global θi = 0.01 * randn(rng, n_weights)
    global iter = 0
end

reset()

solve_UODE(θi, max_nb_protocols, p_system, tspans, σ0, σ12_all, callback, 
            model_univ, loss_univ, max_nb_iter)
#************************************************************************

# For some reason, the solution is zero after 3-4 iterations. SOMETHING FUNDAMENTALLY WRONG!
#--------------------------------------------------------
#*************************************************
# Prediction using trained weights
# We use protocols  different than those used for training
#*************************************************
v21_1(t) = 2*cos(3*ω*t/4)
v21_2(t) = 2*cos(3*ω*t)
v21_3(t) = 2*cos((3*ω*t)/2)
v21_4(t) = 1.5
v21_fct = Any[v21_1, v21_2, v21_3, v21_4]
n_protocol = 4
protocols, tspans, tsaves = setup_protocols(n_protocol, v21_fct, tspan) 
@show protocols
target_labels = ["σ12a","N1","N2","σ12b"]
# Should be protocol_titles
protocol_labels= ["2cos(3ωt/4)", "2cos(ωt)", "2cos(ωt)", "1.5"]
labels = (target=target_labels, protocol=protocol_labels)

# Function closures. The functions are defined in giesekus_impl.jl
base_model(k) = fct_giesekus(tspans[k], tsaves[k], p_giesekus, σ0, protocols[k])
# Only first two parameters are required
ude_model(k, θ)  = fct_ude(tspans[k], tsaves[k], θ, p_giesekus[1:2], σ0, protocols[k], model_univ)

fcts = (base_model=base_model, ude_model=ude_model)

# Plot the results
plots, halt = my_plot_solution(θ0, θi, protocols, labels, fcts)

new_plot = plot(plots..., layout=(2,2), size=(800,600))
