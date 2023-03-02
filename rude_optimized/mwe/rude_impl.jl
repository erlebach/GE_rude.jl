
function single_run(dct)

	# Define the simple shear deformation protocol (all components except v21(t))
    v11(t) = 0
    
    # Initial conditions and time span
	tspan = (0.0f0, 1.0f0)

	σ0 = SA[0.  0.  0.; 0.  0.  0.; 0.  0.  0.]  # Generates set_index! error
	v21_protoc = [ (t) -> 0. for i in 1:8]

	# Linear array of size 9
	protocols = [ [v11,v11,v11,  v21_protoc[1], v11,v11,v11,v11,v11] ]
    nb_protoc = 1
    
	p_system = Float32[0., 0.]  # change 2023-02-26_15:40
    
	θ0 = [0.] #zeros(size(p_model))
	θi = [0.] #p_model
    
    # Rewrite above section to use Optimizer rather than sciml_train
	σ12_all = [0]  # actually solution to Giesekus

	k = 1
	# Loss function closure (first parameter: concatenate all parameters)
        loss_fn(θ) = loss_univ([θ; p_system], protocols[1], tspan, σ0, σ12_all, 1, dct)
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p)->loss_fn(x),  adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)  
        parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=dct[:maxiters]) 
    #end
end
