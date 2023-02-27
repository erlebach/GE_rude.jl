
# For more control, this function could be a Module. But why polute the namespace?


# global scope (in Main, or in any Module)
# The use of "let" allows the creation of static variables similar to C++. The objective
# is that count1 represents the number of times the callback function was called. The iteration
# count is initialied to iter=0, and cannot be changed without reinitialzing the static variable `count`
# which may or may not be possible.
# To make this work the callback function must be defined in the global environment, and therefore, 
# could be included in rude_functions. However, I consider it more closely related to the implementation, 
# so I leave it in this file. It is a choice that has no perfect answer. 

# I would like to be able to restart the simulation without reloading the model and initializing the iteration count. 
# How to do that? 

iter = 0
let count1=iter
    global callback_static
    function callback_static(θ, l, protocols, tspans, σ0, σ12_all, trajectories, iter)
		push!(dct[:losses], l)
        count1 += 1
        println("callback_static, iter: $count1, loss: $l")
        return false
    end
end

callback = callback_static

function plot_callback(θ, l, θ0, σ0, p_giesekus, step; dct)
	tspan = (0., dct[:T])
	# do not exist the optimization code that calls the callback
    halt = false  
	push!(dct[:losses], l)
	nb_iter = length(dct[:losses])
	println("callback_static, iter: $nb_iter, loss: $l")
	dct[:nb_iter_optim] = nb_iter
    
    if nb_iter % step == 0
        plots, halt = plot_solution(θ0, θ, tspan, σ0, p_giesekus; dct)
        callback_plot = plot(plots..., plot_title="iteration $nb_iter, $(now())") 
		display(callback_plot)
        fig_file_name = "callback_plot_$nb_iter.pdf"
        savefig(callback_plot, fig_file_name)
		println("==> after display plot, iter: $nb_iter")
    end
    return halt
end

function single_run(dct)

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
	tspan = (0.0f0, dct[:T])
    tsave = range(tspan[1],tspan[2],length=50)
    σ0 = [0f0,0f0,0f0,0f0,0f0,0f0]
    
    # Build the protocols for different 'experiments'. For example,
    # γ₀ = [.01, .1, 1., 10.]
    # γ̇ = γ₀ω cos(ωt)
    # v21_1 = (∇v)_)(21)  (_1 means first protocol of an ensemble of 8 protocols)
    #    = the derivative of v₂ with respect to x₁. 
    # v21 = γ̇
    ω = dct[:ω]  # may lead to "Warning: Reverse-Mode AD VJP choices all failed. Falling back to numerical VJPs"
    γ = dct[:γ]
    γs = dct[:γ_protoc]
    ωs = dct[:ω_protoc]

    # Modify the choice of protcol from `rude-script.jl`

    ##
    v21_1(t) = γs[1]*γ*cos(ωs[1]*ω*t)
    v21_2(t) = γs[2]*γ*cos(ωs[2]*ω*t)
    v21_3(t) = γs[3]*γ*cos(ωs[3]*ω*t)
    v21_4(t) = γs[4]*γ*cos(ωs[4]*ω*t)
    v21_5(t) = γs[5]*γ*cos(ωs[5]*ω*t)
    v21_6(t) = γs[6]*γ*cos(ωs[6]*ω*t)
    v21_7(t) = γs[7]*γ*cos(ωs[7]*ω*t)
    v21_8(t) = γs[8]*γ*cos(ωs[8]*ω*t)
    ##

    #=
    v21_1(t) = γs[1]*γ*cos(ωs[1]*ω*t)
    v21_2(t) = γs[2]*γ*cos(ωs[2]*ω*t)
    v21_3(t) = γs[3]*γ*cos(ωs[3]*ω*t)
    v21_4(t) = γs[4]*γ*cos(ωs[4]*ω*t)
    v21_5(t) = γs[5]*γ*cos(ωs[5]*ω*t)
    v21_6(t) = γs[6]*γ*cos(ωs[6]*ω*t)
    v21_7(t) = γs[7]*γ*cos(ωs[7]*ω*t)
    v21_8(t) = γs[8]*γ*cos(ωs[8]*ω*t)
    =#


    #=
    v21_1(t) = 1*cos(ω*t)
    v21_2(t) = 2*cos(ω*t)
    v21_3(t) = 1*cos(ω*t/2)
    v21_4(t) = 2*cos(ω*t/2)
    v21_5(t) = 1*cos(2*ω*t)
    v21_6(t) = 2*cos(2*ω*t)
    v21_7(t) = 1*cos(ω*t/3)
    v21_8(t) = 2*cos(ω*t/3)
    =#

	#=
    # GE: How are v11, etc computed? All others are set to zero.
    gradv_1 = [v11,v12,v13,v21_1,v22,v23,v31,v32,v33]
    gradv_2 = [v11,v12,v13,v21_2,v22,v23,v31,v32,v33]
    gradv_3 = [v11,v12,v13,v21_3,v22,v23,v31,v32,v33]
    gradv_4 = [v11,v12,v13,v21_4,v22,v23,v31,v32,v33]
    gradv_5 = [v11,v12,v13,v21_5,v22,v23,v31,v32,v33]
    gradv_6 = [v11,v12,v13,v21_6,v22,v23,v31,v32,v33]
    gradv_7 = [v11,v12,v13,v21_7,v22,v23,v31,v32,v33]
    gradv_8 = [v11,v12,v13,v21_8,v22,v23,v31,v32,v33]
    protocols = [gradv_1, gradv_2, gradv_3, gradv_4, gradv_5, gradv_6, gradv_7, gradv_8]
    =#

    #v21_protoc = Vector{AbstractFunction}(undef, 8)
    # Generate one function for each protocol
    v21_protoc = [ (t) -> dct[:γ_protoc][i]*cos(dct[:ω_protoc][i]*t) for i in 1:8]

    # GE: How are v11, etc computed? All others are set to zero. 
    gradv_protoc = [ [v11,v12,v13,  v21_protoc[i], v22,v23,v31,v32,v33] for i in 1:8]
    protocols = gradv_protoc  # Should be the same as the original vector of protocols

    nb_protoc = dct[:nb_protocols]
    # I only allowed for 8 protocols (see above)
    nb_protocols = length(protocols)
    if nb_protoc > nb_protocols
        dct[:nb_protocols] = nb_protocols
    end

    tspans = [tspan for i in 1:nb_protocols]
    tsaves = [tsave for i in 1:nb_protocols]
    #tspans = [tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan]
    #tsaves = [tsave, tsave, tsave, tsave, tsave, tsave, tsave, tsave]
    
    protocols = protocols[1:nb_protoc]
    
    #p = (η0 = 1, τ = 1, α = 0.8)
    #function solve_model(p, σ0, protocols, tspans)
    gie = dct[:dct_giesekus]
    η0 = gie[:η0]
    τ = gie[:τ]
    α = gie[:α]
    #η0 = 1
    #τ = 1
    #α = 0.8
    p_giesekus = [η0, τ, α]
    σ12_all = Any[]
    t_all = Any[]
    for k = range(1, length(protocols), step=1)
        dudt!(du,u,p,t) = dudt_giesekus!(du,u,p,t,protocols[k])
        prob_giesekus = ODEProblem(dudt!, σ0, tspans[k], p_giesekus)
		solve_giesekus = solve(prob_giesekus, Rodas4(),saveat=dct[:saveat])
        σ12_data = solve_giesekus[4,:]
        push!(t_all, solve_giesekus.t)
        push!(σ12_all, σ12_data)
    end
    
    # NN model for the nonlinear function F(σ,γ̇)
    dNN = dct[:dct_NN]
    act = dNN[:act]
    hid = dNN[:hid_layer]
    model_univ = FastChain(FastDense(dNN[:in_layer], hid, act),
                        FastDense(hid, hid, act),
                        FastDense(hid, hid, act),  # a second hidden layer
                        FastDense(hid, dNN[:out_layer]))
	dct[:model_univ] = model_univ
    
    # The protocol at which we'll start continuation training
    # (choose start_at > length(protocols) to skip training)
    start_at = 9
    
    # Retrain the network from scratch
    start_at = 1
    
    if start_at > 1
        # Load the pre-trained model if not starting from scratch
        @load "tbnn.bson" θi
        p_model = θi
        n_weights = length(θi)
		dct[:n_weights] = n_weights
    else
        # The model weights are destructured into a vector of parameters
        p_model = initial_params(model_univ)
        n_weights = length(p_model)
        p_model = zeros(n_weights)
		dct[:n_weights] = n_weights
    end
    
    # Parameters of the linear response (η0,τ)
    p_system = Float32[1, 1]
    
    θ0 = zeros(size(p_model))
    θi = p_model
    
    # Rewrite above section to use Optimizer rather than sciml_train
    iter = 0
    θi = zeros(size(p_model))
    out_files = []

    # ISSUE: Initial loss is around 70. WHY? 
    for k = range(start_at, length(protocols), step=1)
        println("protocol k= ", k)
        # Loss function closure
        loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k, dct)
        # Callback function closure
        # k are the trajectories (1:8) 
        cb_fun(θ, loss) = callback(θ, loss, protocols[1:k], tspans[1:k], σ0, σ12_all, k, iter)
		cb_plot(θ, loss) = plot_callback(θ, loss, θ0, σ0, p_giesekus, dct[:skip_factor]; dct=dct)
        cb_all(θ, loss) = function ()
           foreach(f->f(θ, loss), (cb_fun, cb_plot))
           return false
        end
        # Callback function closure
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p)->loss_fn(x),  adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)  # get_parameter_values(nn_eqs)) # nn_eqs???
        #parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), abstol=1.e-6, reltol=1.e-6, callback=cb_plot, maxiters=dct[:maxiters]) # 
		# Original code
        #parameter_res = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_plot, sensealg=ReverseDiffVJP(true), allow_f_increases=false, maxiters=dct[:maxiters]) 
		# default lr is 1.e-3
        parameter_res = Optimization.solve(optprob, Optimisers.Adam(1f-3), callback=cb_plot, sensealg = ReverseDiffVJP(true), allow_f_increases=true, maxiters=dct[:maxiters])
        θi = parameter_res.u
        push!(out_files, "tbnn_k=" * string(k))
        #@save "tbnn.bson" θi
        @save out_files[end] θi
    end
    
    # Build full parameter vectors for model testing
    θ0 = [θ0; p_system]  # create single column vector
    θi = [θi; p_system]
    plots, halt = plot_solution(θ0, θi, tspan, σ0, p_giesekus; dct)
	final_plot = plot(plots..., plot_title="Last training step, $(now())")
    display(final_plot)
    return final_plot 
end
