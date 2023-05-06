#savedir = "/Users/alex/Documents/Important/College/FSU Misc/Masters Project/Project Code/Using Phan-Thien Tanner Data as Training Data/" # Make sure it has "/" at the end!
#cd(savedir)

using Flux, Optimization, OptimizationOptimisers, SciMLSensitivity, DifferentialEquations
using Zygote, Plots, LaTeXStrings, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using BSON: @save, @load

#### SET VARIABLES ####
const max_nb_protocols::Int32 = 8
const max_nb_iterations::Int32 = 200
# const max_nb_iterations::Int32 = 5

# The protocol at which we'll start continuation training
# (choose start_at > length(protocols) to skip training)
#start_at = 9 # Don't train
start_at = 1 # Train from scratch
start_at = max_nb_protocols+1 # Read the saved data file
#######################

function write_files(files)
    # Open and close a file for writing. This should empty any 
    # existing file
    for fn in files
        file = open(fn, "w")
        close(file)
    end
end

write_files(["PTT_lambda.txt", "PTT_g.txt"])

# TODO: Actually change from Giesekus to PTT
function dudt_PTT!(du, u, p, t, gradv)

    # Destructure the parameters
    η0, τ, α, G, ϵ = p
    
    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u
    
    # Specify the velocity gradient tensor
    v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv
    
    # Nonlinear term in PTT Model
    fPTT = exp((ϵ/G)*(σ11+σ22+σ33))

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)
    γd22 = 2*v22(t)
    γd33 = 2*v33(t)
    γd12 = v12(t) + v21(t)
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)
    ω12 = v12(t) - v21(t)
    ω13 = v13(t) - v31(t)
    ω23 = v23(t) - v32(t)

    F11 = -τ*(σ11*γd11 + σ12*γd12 + σ13*γd13)
    F22 = -τ*(σ12*γd12 + σ22*γd22 + σ23*γd23)
    F33 = -τ*(σ13*γd13 + σ23*γd23 + σ33*γd33)
    F12 = -τ*(σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23)/2
    F13 = -τ*(σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33)/2
    F23 = -τ*(σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33)/2

    # The model differential equations
    du[1] = η0*γd11/τ - (σ11*fPTT)/τ - (ω12*σ12 + ω13*σ13) - F11/τ
    du[2] = η0*γd22/τ - (σ22*fPTT)/τ - (ω23*σ23 - ω12*σ12) - F22/τ
    du[3] = η0*γd33/τ - (σ33*fPTT)/τ + (ω13*σ13 + ω23*σ23) - F33/τ
    du[4] = η0*γd12/τ - (σ12*fPTT)/τ - (ω12*σ22 + ω13*σ23 - σ11*ω12 + σ13*ω23)/2 - F12/τ
    du[5] = η0*γd13/τ - (σ13*fPTT)/τ - (ω12*σ23 + ω13*σ33 - σ11*ω13 - σ12*ω23)/2 - F13/τ
    du[6] = η0*γd23/τ - (σ23*fPTT)/τ - (ω23*σ33 - ω12*σ13 - σ12*ω13 - σ22*ω23)/2 - F23/τ
end

function tbnn(σ,γd,model_weights, t, lambda_file, g_file, F_file, σ_file)
# function tbnn(σ,γd,model_weights,printout)
    # Tensor basis neural network (TBNN)
    # Unpack the inputs
    σ11,σ22,σ33,σ12,σ13,σ23 = σ
    γd11,γd22,γd33,γd12,γd13,γd23 = γd

    # Compute elements of the tensor basis
    # T1 = I, T2 = σ, T3 = γd
    # T4 = σ⋅σ
    T4_11 = σ11^2 + σ12^2 + σ13^2
    T4_22 = σ12^2 + σ22^2 + σ23^2
    T4_33 = σ13^2 + σ23^2 + σ33^2
    T4_12 = σ11*σ12 + σ12*σ22 + σ13*σ23
    T4_13 = σ11*σ13 + σ12*σ23 + σ13*σ33
    T4_23 = σ12*σ13 + σ22*σ23 + σ23*σ33

    # T5 = γd⋅γd
    T5_11 = γd11^2 + γd12^2 + γd13^2
    T5_22 = γd12^2 + γd22^2 + γd23^2
    T5_33 = γd13^2 + γd23^2 + γd33^2
    T5_12 = γd11*γd12 + γd12*γd22 + γd13*γd23
    T5_13 = γd11*γd13 + γd12*γd23 + γd13*γd33
    T5_23 = γd12*γd13 + γd22*γd23 + γd23*γd33

    # T6 = σ⋅γd + γd⋅σ
    T6_11 = 2*(σ11*γd11 + σ12*γd12 + σ13*γd13)
    T6_22 = 2*(σ12*γd12 + σ22*γd22 + σ23*γd23)
    T6_33 = 2*(σ13*γd13 + σ23*γd23 + σ33*γd33)
    T6_12 = σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23
    T6_13 = σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33
    T6_23 = σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33

    # T7 = σ⋅σ⋅γd + γd⋅σ⋅σ
    T7_11 = 2*(T4_11*γd11 + T4_12*γd12 + T4_13*γd13)
    T7_22 = 2*(T4_12*γd12 + T4_22*γd22 + T4_23*γd23)
    T7_33 = 2*(T4_13*γd13 + T4_23*γd23 + T4_33*γd33)
    T7_12 = T4_11*γd12 + T4_12*γd22 + T4_13*γd23 + γd11*T4_12 + γd12*T4_22 + γd13*T4_23
    T7_13 = T4_11*γd13 + T4_12*γd23 + T4_13*γd33 + γd11*T4_13 + γd12*T4_23 + γd13*T4_33
    T7_23 = T4_12*γd13 + T4_22*γd23 + T4_23*γd33 + γd12*T4_13 + γd22*T4_23 + γd23*T4_33

    # T8 = σ⋅γd⋅γd + γd⋅γd⋅σ
    T8_11 = 2*(σ11*T5_11 + σ12*T5_12 + σ13*T5_13)
    T8_22 = 2*(σ12*T5_12 + σ22*T5_22 + σ23*T5_23)
    T8_33 = 2*(σ13*T5_13 + σ23*T5_23 + σ33*T5_33)
    T8_12 = σ11*T5_12 + σ12*T5_22 + σ13*T5_23 + T5_11*σ12 + T5_12*σ22 + T5_13*σ23
    T8_13 = σ11*T5_13 + σ12*T5_23 + σ13*T5_33 + T5_11*σ13 + T5_12*σ23 + T5_13*σ33
    T8_23 = σ12*T5_13 + σ22*T5_23 + σ23*T5_33 + T5_12*σ13 + T5_22*σ23 + T5_23*σ33

    # T9 = σ⋅σ⋅γd⋅γd + γd⋅γd⋅σ⋅σ
    T9_11 = 2*(T4_11*T5_11 + T4_12*T5_12 + T4_13*T5_13)
    T9_22 = 2*(T4_12*T5_12 + T4_22*T5_22 + T4_23*T5_23)
    T9_33 = 2*(T4_13*T5_13 + T4_23*T5_23 + T4_33*T5_33)
    T9_12 = T4_11*T5_12 + T4_12*T5_22 + T4_13*T5_23 + T5_11*T4_12 + T5_12*T4_22 + T5_13*T4_23
    T9_13 = T4_11*T5_13 + T4_12*T5_23 + T4_13*T5_33 + T5_11*T4_13 + T5_12*T4_23 + T5_13*T4_33
    T9_23 = T4_12*T5_13 + T4_22*T5_23 + T4_23*T5_33 + T5_12*T4_13 + T5_22*T4_23 + T5_23*T4_33

    # Compute the integrity basis from scalar invariants
    # λ1 = tr(σ)
    λ1 = σ11 + σ22 + σ33

    # λ2 = tr(σ^2)
    λ2 = T4_11 + T4_22 + T4_33

    # λ3 = tr(γd^2)
    λ3 = T5_11 + T5_22 + T5_33

    # λ4 = tr(σ^3)
    λ4 = σ11*T4_11 + σ22*T4_22 + σ33*T4_33 + 2*(σ12*T4_12 + σ13*T4_13 + σ23*T4_23)

    # λ5 = tr(γd^3)
    λ5 = γd11*T5_11 + γd22*T5_22 + γd33*T5_33 + 2*(γd12*T5_12 + γd13*T5_13 + γd23*T5_23)

    # λ6 = tr(σ^2⋅γd^2)
    λ6 = T4_11*T5_11 + T4_22*T5_22 + T4_33*T5_33 + 2*(T4_12*T5_12 + T4_13*T5_13 + T4_23*T5_23)

    # λ7 = tr(σ^2⋅γd)
    λ7 = (T7_11 + T7_22 + T7_33)/2

    # λ8 = tr(σ⋅γd^2)
    λ8 = (T8_11 + T8_22 + T8_33)/2

    # λ9 = tr(σ⋅γd)
    λ9 = (T6_11 + T6_22 + T6_33)/2

    # Run the integrity basis through a neural network
    model_inputs = [λ1;λ2;λ3;λ4;λ5;λ6;λ7;λ8;λ9]
    g1,g2,g3,g4,g5,g6,g7,g8,g9 = re(model_weights)(model_inputs)
    
    if lambda_file != ""
        file = open("$(lambda_file)", "a" )
        println(file, "$t, $λ1, $λ2, $λ3, $λ4, $λ5, $λ6, $λ7, $λ8, $λ9")
        close(file)
    end
    if g_file != ""
        file = open("$(g_file)", "a" )
        println(file, "$t, $g1, $g2, $g3, $g4, $g5, $g6, $g7, $g8, $g9")
        close(file)
    end

    # Tensor combining layer
    F11 = g1 + g2*σ11 + g3*γd11 + g4*T4_11 + g5*T5_11 + g6*T6_11 + g7*T7_11 + g8*T8_11 + g9*T9_11
    F22 = g1 + g2*σ22 + g3*γd22 + g4*T4_22 + g5*T5_22 + g6*T6_22 + g7*T7_22 + g8*T8_22 + g9*T9_22
    F33 = g1 + g2*σ33 + g3*γd33 + g4*T4_33 + g5*T5_33 + g6*T6_33 + g7*T7_33 + g8*T8_33 + g9*T9_33
    F12 = g2*σ12 + g3*γd12 + g4*T4_12 + g5*T5_12 + g6*T6_12 + g7*T7_12 + g8*T8_12 + g9*T9_12
    F13 = g2*σ13 + g3*γd13 + g4*T4_13 + g5*T5_13 + g6*T6_13 + g7*T7_13 + g8*T8_13 + g9*T9_13
    F23 = g2*σ23 + g3*γd23 + g4*T4_23 + g5*T5_23 + g6*T6_23 + g7*T7_23 + g8*T8_23 + g9*T9_23

    if F_file != ""
        file = open("$(F_file)", "a" )
        println(file, "$t, $F11, $F22, $F33, $F12, $F13, $F23")
        close(file)
    end
    if σ_file != ""
        file = open("$(σ_file)", "a" )
        println(file, "$t, $σ11, $σ22, $σ33, $σ12, $σ13, $σ23")
        close(file)
    end

    # if printout == true
    #     # Not too good with Julia printing format
    #     num_digits = 4
    #     print(round(λ1, digits=num_digits))
    #     print(" ")
    
    return F11,F22,F33,F12,F13,F23
end

# Notice the semicolon (;) after required arguments. What follows are optional arguments. 
# The semi-colon is not required when calling the function.
function dudt_univ!(du, u, p, t, gradv; lambda_file="", g_file="", F_file="", σ_file="")
    # Destructure the parameters
    model_weights = p[1:n_weights]
    η0 = p[end - 1]
    τ = p[end]

    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u

    # Specify the velocity gradient tensor
    v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)
    γd22 = 2*v22(t)
    γd33 = 2*v33(t) 
    γd12 = v12(t) + v21(t)
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)

    # Run stress/strain through a TBNN
    γd = [γd11,γd22,γd33,γd12,γd13,γd23]
	F11,F22,F33,F12,F13,F23 = tbnn(u,γd,model_weights,t, lambda_file, g_file, F_file, σ_file)

    # The model differential equations
    dσ11 = η0*γd11/τ - σ11/τ + 2*v11(t)*σ11 + v21(t)*σ12 + v31(t)*σ13 + σ12*v21(t) + σ13*v31(t) - F11/τ
    dσ22 = η0*γd22/τ - σ22/τ + 2*v22(t)*σ22 + v12(t)*σ12 + v32(t)*σ23 + σ12*v12(t) + σ23*v32(t) - F22/τ
    dσ33 = η0*γd33/τ - σ33/τ + 2*v33(t)*σ33 + v13(t)*σ13 + v23(t)*σ23 + σ13*v13(t) + σ23*v23(t) - F33/τ
    dσ12 = η0*γd12/τ - σ12/τ + v11(t)*σ12 + v21(t)*σ22 + v31(t)*σ23 + σ11*v12(t) + σ12*v22(t) + σ13*v32(t) - F12/τ
    dσ13 = η0*γd13/τ - σ13/τ + v11(t)*σ13 + v21(t)*σ23 + v31(t)*σ33 + σ11*v13(t) + σ12*v23(t) + σ13*v33(t) - F13/τ
    dσ23 = η0*γd23/τ - σ23/τ + v12(t)*σ13 + v22(t)*σ23 + v32(t)*σ33 + σ12*v13(t) + σ22*v23(t) + σ23*v33(t) - F23/τ

    # Update in place
    du[1] = dσ11
    du[2] = dσ22
    du[3] = dσ33
    du[4] = dσ12
    du[5] = dσ13
    du[6] = dσ23
end

function ensemble_solve(θ,ensemble,protocols,tspans,σ0,trajectories)
	# Define the (default) ODEProblem
	dudt_protocol!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[1])
	prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

	# Remake the problem for different protocols
	function prob_func(prob, i, repeat)
		dudt_remade!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[i])
		remake(prob, f=dudt_remade!, tspan=tspans[i])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories, 
		    sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)
end

function loss_univ(θ,protocols,tspans,σ0,σ12_all,trajectories)
	loss = 0
	results = ensemble_solve(θ,EnsembleThreads(),protocols,tspans,σ0,trajectories)
	for k = range(1,trajectories,step=1)
		σ12_pred = results[k][4,:]
		σ12_data = σ12_all[k]
		loss += sum(abs2, σ12_pred - σ12_data)
	end
	loss += 0.01*norm(θ,1)
	return loss
end

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
σ0 = [0f0,0f0,0f0,0f0,0f0,0f0]

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

# Solve for the PTT model
η0 = 1
τ = 1
α = 0.8
G = 1
ϵ = 1 # 0.1
p_PTT = [η0,τ,α,G,ϵ]
σ12_all = Any[]
t_all = Any[]
#for k = range(1,length(protocols),step=1)
for k = range(1,max_nb_protocols)  # temporary to save time. GE
	dudt!(du,u,p,t) = dudt_PTT!(du,u,p,t,protocols[k])
	prob_PTT = ODEProblem(dudt!, σ0, tspans[k], p_PTT)
	solve_PTT = solve(prob_PTT,Rodas4(),saveat=0.2)
	σ12_data = solve_PTT[4,:]
	push!(t_all, solve_PTT.t)
	push!(σ12_all, σ12_data)
end

# NN model for the nonlinear function F(σ,γ̇)
nb_hidden = 32
model_univ = Flux.Chain(Flux.Dense(9, nb_hidden, tanh),
                       Flux.Dense(nb_hidden, nb_hidden, tanh),
                       Flux.Dense(nb_hidden, 9))
p_model, re = Flux.destructure(model_univ)

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
    print("iter: ")
    print(iter)
    print(" ==> loss: ")
    println(l)
    return false
end

# Continutation training loop
adtype = Optimization.AutoZygote()
# for k = range(start_at,length(protocols),step=1)
for k = range(1,max_nb_protocols)  # temporary to save time. GE
    println("==> k= $k")
	loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k)
	cb_fun(θ, l) = callback(θ, l, protocols[1:k], tspans[1:k], σ0, σ12_all, k)
	optf = Optimization.OptimizationFunction((x,p) -> loss_fn(x), adtype)
	optprob = Optimization.OptimizationProblem(optf, θi)
	result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback = cb_fun, maxiters = max_nb_iterations)
	global θi = result_univ.u
    #  TODO: save protocol k into its own "tbnn$k.bson" file
	@save "tbnn.bson" θi
end

# Write weights to text files for OpenFOAM 
# writedlm("OpenFOAM/RUDE/weights1.txt", θi[1:10*32])
# writedlm("OpenFOAM/RUDE/weights2.txt", θi[10*32 + 1:10*32 + 33*32])
# writedlm("OpenFOAM/RUDE/weights3.txt", θi[10*32 + 33*32 + 1:end])

# Build full parameter vectors for model testing
θ0 = [θ0; p_system]
θi = [θi; p_system]

# Test the UDE on a new condition
v21_1(t) = 2*cos(3*ω*t/4)
v21_2(t) = 2*cos(3*ω*t)
v21_3(t) = 2*cos(1.5*ω*t)
v21_4(t) = 1.5f0
gradv_1 = [v11,v12,v13,v21_1,v22,v23,v31,v32,v33]
gradv_2 = [v11,v12,v13,v21_2,v22,v23,v31,v32,v33]
gradv_3 = [v11,v12,v13,v21_3,v22,v23,v31,v32,v33]
gradv_4 = [v11,v12,v13,v21_4,v22,v23,v31,v32,v33]
protocols = [gradv_1, gradv_2, gradv_3, gradv_4]
target = ["σ12","N1","N2","ηE"]
xlimits = (0,12)
tspan = (0.0f0, 12.0f0)
plots = []
legend_present = false

# Store diffent targets and protocols into different files

for jj = range(1,length(target),step=1)
    for k = range(1,length(protocols),step=1)
        g_file = "PTT_g_target$(jj)_protocol$(k).txt"
        lambda_file = "PTT_lambda_target$(jj)_protocol$(k).txt"
        F_file = "PTT_F_target$(jj)_protocol$(k).txt"
        σ_file = "PTT_σ_target$(jj)_protocol$(k).txt"

        if (jj == 1) && (k == 1)
            legend_present = true
        else
            legend_present = false
        end
        # Solve the PTT model
        dudt!(du,u,p,t) = dudt_PTT!(du,u,p,t,protocols[k])
        local prob_PTT = ODEProblem(dudt!, σ0, tspan, p_PTT)
        local solve_PTT = solve(prob_PTT,Tsit5(),saveat=0.1)
        local σ12_data = solve_PTT[4,:]
        N1_data = solve_PTT[1,:] - solve_PTT[2,:]
        N2_data = solve_PTT[2,:] - solve_PTT[3,:]

        # Solve the UDE pre-training
        dudt_ude!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[k])
        local prob_univ = ODEProblem(dudt_ude!, σ0, tspan, θ0)
        local sol_pre = solve(prob_univ, Tsit5(),abstol = 1e-8, reltol = 1e-6, saveat=0.1)
        σ12_ude_pre = sol_pre[4,:]
        N1_ude_pre = sol_pre[1,:] - sol_pre[2,:]
        N2_ude_pre = sol_pre[2,:] - sol_pre[3,:]
        
        # Solve the UDE post-training
		# Notice the last two arguments
        println("******* $g_file, $lambda_file, $F_file, $σ_file")
        dudt_ude!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[k], 
            lambda_file=lambda_file, g_file=g_file, F_file=F_file, σ_file=σ_file)
        prob_univ = ODEProblem(dudt_ude!, σ0, tspan, θi)
        sol_univ = solve(prob_univ, Tsit5(),abstol = 1e-8, reltol = 1e-6, saveat=0.1)
        σ12_ude_post = sol_univ[4,:]
        N1_ude_post = sol_univ[1,:] - sol_univ[2,:]
        N2_ude_post = sol_univ[2,:] - sol_univ[3,:]

        # Plot
        plot(xlabel=L"t/\tau_1", xlims=xlimits)
        if target[jj] == "σ12"
            plot!(solve_PTT.t[1:2:end],σ12_data[1:2:end], c=:red, linestyle=:solid, ms=2, label="PTT", gridlinewidth=1, legend=:topright) 
            plot!(sol_pre.t,σ12_ude_pre,lw=1, c=:blue, ls=:dash, title=L"σ_{12}"*", Protocol "*string(k), label="UDE Pre") 
            plot_σ12 = plot!(sol_univ.t,σ12_ude_post,lw=1, c=:black, label="UDE Post", m=:o, ms=2, t=:scatter, legend=legend_present, ylabel=L"\sigma_{12}/G_0") 
            # if (k <= 2) # Protocols 1 and 2
            # if (k > 2) # Protocols 3 and 4
            if (true) # All protocols
                push!(plots, plot_σ12)
            end
        elseif target[jj] == "N1"
            plot!(solve_PTT.t[1:2:end],N1_data[1:2:end], linestyle=:solid, ms=2, c=:red, label="PTT", gridlinewidth=1)
            plot!(sol_pre.t,N1_ude_pre, lw=1, c=:blue, ls=:dash, title=L"N_1"*", Protocol "*string(k), label="UDE Pre")  
            plot_N1 = plot!(sol_univ.t,N1_ude_post,lw=1, c=:black, label="UDE Post", m=:o, ms=2, t=:scatter, legend=legend_present, ylabel=L"N_1/G_0") 
            # if (k <= 2) # Protocols 1 and 2
            # if (k > 2) # Protocols 3 and 4
            if (true) # All protocols
                push!(plots, plot_N1)
            end
        elseif target[jj] == "N2"
            plot!(solve_PTT.t,N2_data, linestyle=:solid, c=:red, ms=2, label="PTT", gridlinewidth=1, legend=:topright) 
            plot!(sol_pre.t,N2_ude_pre,lw=1, c=:blue, title=L"N_2"*", Protocol "*string(k), ls=:dash, label="UDE Pre") 
            plot_N2 = plot!(sol_univ.t,N2_ude_post,lw=1, label="UDE Post", c=:black, m=:o, ms=2, t=:scatter, legend=legend_present, ylabel=L"N_2/G_0") 
            # if (k <= 2) # Protocols 1 and 2
            # if (k > 2) # Protocols 3 and 4
            if (true) # All protocols
                push!(plots, plot_N2)
            end
        elseif target[jj] == "ηE"
            plot!(solve_PTT.t,-N2_data-N1_data, c=:red, linestyle=:solid, ms=2, label="PTT", gridlinewidth=1, legend=:bottomright) 
            plot!(sol_pre.t,-N2_ude_pre-N1_ude_pre, lw=1, c=:blue, title=L"η_E"*", Protocol "*string(k), ls=:dashdot, label="UDE Pre") 
            plot_N2N1 = plot!(sol_univ.t,-N2_ude_post-N1_ude_post, lw=0.5, c=:black, label="UDE Post", m=:o, ms=2, t=:scatter, legend=legend_present, ylabel=L"\eta_E/G_0") 
            # if (k <= 2) # Protocols 1 and 2
            # if (k > 2) # Protocols 3 and 4
            if (true) # All protocols
                push!(plots, plot_N2N1)
            end
        end
        # ax.set_xlim([0,24])
        # ax[:tick_params](axis="both", direction="in", which="both", right="true", top="true", labelsize=12)
        # savefig("test"*string(k)*".pdf")
    end
end
# final_plot = plot(plots..., layout=(4,2), size = (1000,1000)) # 2 protocols
final_plot = plot(plots..., layout=(4,4), size = (1000,1000)) # All protocols
savefig(final_plot, "PTT_final_plot.pdf")
plots = [] # Reset plot list

# Reset below before rerunning code
t_all = []
σ12_all = []

# I have file with numbers. I'd like to identify the first line that starts with 12.0, skip the next line, and 
# output the remaining lines to a new file. 
function get_data(filename)
    # Read in file
    file = open(filename)
    lines = readlines(file)
    close(file)
    
    # Find the first line that starts with 12.0
    for i in 1:length(lines)
        if occursin("12.0", lines[i])
            start = i
            break
        end
    end
    
    # Skip the next line
    stop = start + 2
    
    # Output the remaining lines to a new file
    file = open(filename)
    lines = readlines(file)
    close(file)
    lines = lines[stop:end]
    file = open(filename, "w")
    for i in 1:length(lines)
        write(file, lines[i])
        write(file, "\n")
    end
    close(file)
end