
function single_run(dct)

	# Define the simple shear deformation protocol (all components except v21(t))
    v11(t) = 0
    
    # Initial conditions and time span
	tspan = (0.0f0, dct[:T])
    tsave = range(tspan[1],tspan[2],length=50)

	#σ0 = SA_F32[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]  # Generates load error

	σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]  # Generates set_index! error
	#setindex!(::StaticArraysCore.SMatrix{3, 3, Float64, 9}, value, ::Int) is not defined. 
	#
	println("typeof σ0: ", typeof(σ0))
    
    ω = dct[:ω]  
    γ = dct[:γ]
    γs = dct[:γ_protoc]
    ωs = dct[:ω_protoc]

    # Modify the choice of protcol from `rude-script.jl`

    v21_1(t) = γs[1] * γ * cos(ωs[1] *ω * t)

    # Generate one function for each protocol
	v21_protoc = [ (t) -> γs[i] * cos(ωs[i]*t) for i in 1:8]

    # GE: How are v11, etc computed? All others are set to zero. 
	# Linear array of size 9
	protocols = [ [v11,v11,v11,  v21_protoc[1], v11,v11,v11,v11,v11] ]
    nb_protoc = 1
    # I only allowed for 8 protocols (see above)
    nb_protocols = length(protocols)
    dct[:nb_protocols] = nb_protoc

    tspans = [tspan for i in 1:nb_protocols]
    tsaves = [tsave for i in 1:nb_protocols]
    
    protocols = protocols[1:nb_protoc]
    
    #p = (η0 = 1, τ = 1, α = 0.8)
    #function solve_model(p, σ0, protocols, tspans)
    gie = dct[:dct_giesekus]
    η0 = gie[:η0]
    τ = gie[:τ]
    α = gie[:α]
    p_giesekus = SA[η0, τ, α]
    σ12_all = []
    t_all = []
    
    # NN model for the nonlinear function F(σ,γ̇)

	#----------------------- MODEL ---------
	##
    model_univ = FastChain(FastDense(9, 1, identity),
						FastDense(1, 9))
	dct[:model_univ] = model_univ
	##
	#--------------------------------------
    
    # The protocol at which we'll start continuation training
	# (choose start_at > length(protocols) to skip training) (start_at > 1)
	# Retrain the network from scratch (start_at = 1)
	start_at = 1
    
	##
        # The model weights are destructured into a vector of parameters
		println("Train the model from scratch")
        p_model = initial_params(model_univ)
        n_weights = length(p_model)
        p_model = zeros(n_weights)
	##

	dct[:n_weights] = n_weights
	dct[:model_weights] = p_model
    
    # Parameters of the linear response (η0,τ)
    gie = dct[:dct_giesekus]
    η0 = gie[:η0]
    τ = gie[:τ]
	p_system = Float32[η0, τ]  # change 2023-02-26_15:40
    
    θ0 = zeros(size(p_model))
    θi = p_model
    
    # Rewrite above section to use Optimizer rather than sciml_train
    iter = 0  # SHOULD BE the last iteration run previously
	σ12_all = 0  # actually solution to Giesekus

	k = 1
	# Loss function closure (first parameter: concatenate all parameters)
        loss_fn(θ) = loss_univ([θ; p_system], protocols[1:1], tspans[1:1], σ0, σ12_all, 1, dct)
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p)->loss_fn(x),  adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)  
        parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=dct[:maxiters]) 
    #end
end
