using Revise

# Use a module to avoid namespace pollution
# The objective is to run a series of experiments with different parameters

using DiffEqFlux, Flux, Optim, DifferentialEquations, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Zygote
using Plots
using DataFrames, CSV, YAML
using BSON: @save, @load
using Dates

# Need a global function
include("rude_functions.jl")
include("myplots.jl")
include("rude_impl.jl")

# There are 8 protocols for each value of ω0 and γ0. So if you have 4 pairs (ω0, γ0), 
# you will have 8 protocols for each pair. (That is not what Sachin's problem proposes). 
# So perhaps one has to generalize how the protocols are stored in the main program to run 
# a wider variety of tests. Ask questions, Alex. 

print("after includes: ", now())

function setup_dict_production()
    dct_params = Dict()
    dct_params[:ω0] = [100.f0]
    dct_params[:ω0] = [1f0]
    dct_params[:γ0] = [.1f0] # not used


    # Create a Dictionary with all parameters
    dct = Dict()
    # Set to lower number to run faster and perhaps get less accurate results
    dct[:maxiters] = 200 # 200  # 200 was number in original code
    # Set to 10 once the program works. More cycles, more oscillations
    dct[:Ncycles] = 3
    dct[:nb_pts_per_cycle] = 80  # Perhaps increase this will increase accuracy? 
    dct[:nb_protocols] = 8  # set to 1 to run faster. Set to 8 for more accurate results
    dct[:skip_factor] = 100
    dct[:dct_giesekus] = Dict()
    # Next two lines are not used yet
    dct[:γ_protoc] = convert(Vector{Float32}, [1, 2, 1, 2, 1, 2, 1, 2])
    dct[:ω_protoc] = convert(Vector{Float32}, [1, 1, 0.5, 0.5, 2., 2., 1/3., 1/3.])
    gie = dct[:dct_giesekus]
    gie[:η0] = 1.f0
    gie[:α] = 0.8f0 # a change to this value propagates correctly to dct[:dct_giesekus][α]
    gie[:τ] = 1.0f0

    dct[:dct_NN] = Dict()
    dNN = dct[:dct_NN]
    dNN[:nb_hid_layers] = 2
    dNN[:in_layer] = 9
    dNN[:out_layer] = 9
    dNN[:hid_layer] = 32
    dNN[:act] = tanh
	dct[:losses] = []    # Accumulate losses
    return dct, dct_params
end

function setup_dict_testing()
    dct, dct_params = setup_dict_production()
    dct_params[:ω0] = [.1f0]
    dct_params[:γ0] = [.1f0] # not used
    dct[:maxiters] = 200
    dct[:nb_protocols] = 1
    dct[:skip_factor] = 10
    dNN = dct[:dct_NN]
    dNN[:nb_hid_layers] = 2
    dNN[:in_layer] = 9
    dNN[:out_layer] = 9
    dNN[:hid_layer] = 32
    dNN[:act] = tanh
    return dct, dct_params
end

# ============== END DICTIONARY DEFINITIONS ==================
production = true  # set to false when testing the code, for faster calculations

if production == true
    # Production: parameters set to generate converged solutions
    dct, dct_params = setup_dict_production()
else 
    # testing: # Parameters set for code to run fast for testing. Results not important. 
    dct, dct_params = setup_dict_testing()  
end

function nearest_multiple_of_10(x::Number)
	return round((x+5) ÷ 10) * 10
end


# Not a good idea to let global variables lying around
# Write dictionary to a database
dicts = []
run = 0

println(".... Start parameter loop ...., now: ", now())
for o in dct_params[:ω0]
    for g in dct_params[:γ0]
        global run += 0
        println("..... run = $(run)")
		dct[:start_datetime] = now()
		dct[:end_datetime] = "Simulation not ended" 
        dct[:datetime] = now()
        dct[:run] = run
        dct[:ω] = o
        dct[:γ] = g
		# dct[:T] is Float64 (but not used in gradient calculation)

		dct[:T] = (2. * π / dct[:ω]) * dct[:Ncycles]  |> nearest_multiple_of_10 |> Float32
		dct[:saveat] = dct[:T] / dct[:nb_pts_per_cycle] / dct[:Ncycles] |> Float32 # where to save the solution

		dct[:T] = 12.f0  #
		dct[:saveat] = 0.1 #  (120 points over a timespan of 12)

		# Trick to make sure that saveat is always a submultiple of the time span. Not used. 
        # dct[:saveat] = (saveat=saveat[2:end-1], save_start=true, save_end=true)
        # dct[:saveat] = (saveat=saveat[1:end], save_start=true, save_end=false)
		# Add these arguments to a call to solve(....; dct[:saveat]...)
		println("Simulation time: ", dct[:T])
		println("saveat time: ", dct[:saveat])

		YAML.write_file("latest_dict_run$run.yml", dct)
		print("before single_run, now: ", now())
        figure = single_run(dct)
        # deepcopy will make sure that the results is different than dct
        # Without it, the dictionaries saved in the list will all be the same
		dct[:end_datetime] = now() # to measure computational time
        push!(dicts, deepcopy(dct))
        fig_file_name = "plot_" * "run$(dct[:run]).pdf"
        savefig(figure, fig_file_name)
    end
end

# Write all dictionaries to a file in YAML format (look it up)
# This file should be saved, together with jl files, plot files, to a folder, for safekeeping
YAML.write_file("dicts.yml", dicts)

