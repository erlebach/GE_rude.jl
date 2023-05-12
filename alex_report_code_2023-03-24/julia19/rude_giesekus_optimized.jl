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

# Set up Parameters
VERBOSE::Bool = false 
const max_nb_protocols::Int32 = 3
const max_nb_iter::Int32 = 5   # 2023-05-10_18:26, why am I not getting 100 iterations per protocol?
start_at = 1 # Train from scratch

n_protocol_train = 8
tspan = (0., 12.)
ampl =  SizedVector{n_protocol_train}([1, 2, 1, 2, 1, 2, 1, 2])
freq =  SizedVector{n_protocol_train}([1., 1., 0.5, 0.5, 2., 2., 1/3, 1/3])
ω = 1.0
v21_fct = Any[t -> ampl[i] * cos(freq[i]*ω*t) for i in range(1, n_protocol_train, step=1)]

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
model_univ = NeuralNetwork(nb_in=9, nb_out=9, layer_size=8, nb_hid_layers=0)
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
  println("===> Loss($iter): $(round(l, digits=4))")
  return false
end

#********************************
# Solve the UODE                
#********************************
solve_UODE(θi, max_nb_protocols, p_system, tspans, σ0, σ12_all, callback, 
            model_univ, loss_univ, model_weights, max_nb_iter)

#--------------------------------------------------------
#*************************************************
# Prediction using trained weights
# We use protocols  different than those used for training
#*************************************************
v21_1(t) = 2*cos(3*ω*t/4)
v21_2(t) = 2*cos(3*ω*t)
v21_3(t) = 2*cos((3*ω*t)/2)
v21_4(t) = 1.5f0
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
ude_model(k, θ)  = fct_ude(tspans[k], tsaves[k], θ, p_giesekus, σ0, protocols[k], model_univ, model_weights)
fcts = (base_model=base_model, ude_model=ude_model)

# Plot the results
plots, halt = my_plot_solution(θ0, θi, protocols, labels, fcts)

new_plot = plot(plots..., layout=(2,2), size=(800,600))

#-------------------------------------------------------
#= 
An Alternative to plotting the solution would be to save the data
to files and have python plot the solution. This is probably wiser so 
that Julia can concentrate on the computationally intensive tasks. 
The data can be made available to collaborators and not require them 
to run the Julia code. 
=#

# Build full parameter vectors for model testing
# θ0: initial network parameters
# θi: final network parameters
# p_system: base model parameters used in UODE
θ0 = [θ0; p_system]
θi = [θi; p_system]

# Test the UDE on a new condition
target = ["σ12","N1","N2","ηE"]
xlimits = (0,24)
tspan = (0.0f0, 24.0f0)
plots = []
legend_present = false

VERBOSE = true # print out λ, g

for jj = range(1,length(target),step=1)
    for k = range(1,length(protocols),step=1)
        if (jj == 1) && (k == 1)
            legend_present = true
        else
            legend_present = false
        end
        # Solve the Giesekus model
        dudt!(du,u,p,t) = dudt_giesekus!(du,u,p,t,protocols[k])
        local prob_giesekus = ODEProblem(dudt!, σ0, tspan, p_giesekus)
        local solve_giesekus = solve(prob_giesekus,Tsit5(),saveat=0.1)
        local σ12_data = solve_giesekus[4,:]
        N1_data = solve_giesekus[1,:] - solve_giesekus[2,:]
        N2_data = solve_giesekus[2,:] - solve_giesekus[3,:]

        # Solve the UDE pre-training
        dudt_ude!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[k])
        local prob_univ = ODEProblem(dudt_ude!, σ0, tspan, θ0)
        local sol_pre = solve(prob_univ, Tsit5(),abstol = 1e-8, reltol = 1e-6, saveat=0.1)
        σ12_ude_pre = sol_pre[4,:]
        N1_ude_pre = sol_pre[1,:] - sol_pre[2,:]
        N2_ude_pre = sol_pre[2,:] - sol_pre[3,:]

        # Solve the UDE post-training
        #  Should θi containe ODE params, or should I write [θi; p_giesekus]?
        prob_univ = ODEProblem(dudt_ude!, σ0, tspan, θi)
        sol_univ = solve(prob_univ, Tsit5(),abstol = 1e-8, reltol = 1e-6, saveat=0.1)
        σ12_ude_post = sol_univ[4,:]
        N1_ude_post = sol_univ[1,:] - sol_univ[2,:]
        N2_ude_post = sol_univ[2,:] - sol_univ[3,:]

        # Plot


dd = function(t) 
    cos(t)
end
ee = function(t) 
    sin(t)
end
ff = (s=dd, c=ee)
ff.s(3.)
ff.s(4.)

println(ff.dd(3.), ff.ee(3.))

yy = (a=3, b=4)
yy.a
yy[:b]
yy = (a=cos, b=sin)
yy.a(3.)
yy.a(2.)