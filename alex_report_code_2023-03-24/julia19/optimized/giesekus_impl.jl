# Date: 2023-05-07
# Author: Gordon Erlebacher, based of of RUDE Julia Implementation

# File to be included by software solving the Giesekus model with a 
# Tensor basis Neural Network

function mat3x3_to_vec6(D)
    d = zeros(6)
    d[1] = D[1,1]
    d[2] = D[2,2]
    d[3] = D[3,3]
    d[4] = D[1,2]
    d[5] = D[1,3]
    d[6] = D[2,3]
    return d
end

function vec6_to_mat3x3(v)
    D = SizedMatrix{3,3}(zeros(3,3))
    D[1,1] = v[1];   D[1,2] = v[4];   D[1,3] = v[5]
    D[2,1] = v[4];   D[2,2] = v[2];   D[2,3] = v[6]
    D[3,1] = v[5];   D[3,2] = v[6];   D[3,3] = v[3]
    return D
end

# This file is included in a file 
function dudt_giesekus_opt!(du, σ, p, t, gradv)
    η0, τ, α = p
    # gradv is an array of functions
    ∇v = SA[0.0 0.0 0.0; gradv[2, 1](t) 0.0 0.0; 0.0 0.0 0.0]
    D = (∇v .+ transpose(∇v))  # necessary to produce same result as original RUDE
    T1 = (η0 / τ) .* D
    T2 = (transpose(∇v) * σ) + (σ * ∇v)
#   coef = α / (τ * η0)
    coef = α / η0
    F = coef * (σ * σ)
    du .= -σ / τ .+ T1 .+ T2 .- F  # 9 equations (static matrix)
end

#---------------------------------------------------------------

fct_giesekus = function (tspan, tsave, p, σ0, protocol)
    # p: parameters for Giesekus base model
    dudt!(du, u, p, t) = dudt_giesekus_opt!(du, u, p, t, protocol)
    prob = ODEProblem(dudt!, σ0, tspan, p)
    sol = solve(prob, Tsit5(), abstol=1.e-6, reltol=1.0e-6, saveat=tsave)
end

fct_ude = function (tspan, tsave, θ, p, σ0, protocol, model_univ)
    # println("fct_ude, len p: $(length(p)),  len θ: $(length(θ))")
    dudt_ude!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocol, model_univ)
    # Solve the UDE 
    prob_ude = ODEProblem(dudt_ude!, σ0, tspan, [θ; p])
    sol_ude = solve(prob_ude, Tsit5(), abstol=1e-6, reltol=1e-6, saveat=tsave)
end

#----------------------------------------------------------------
function dudt_univ_opt!(dσ, σ, p, t, gradv, model_univ)
    # the parameters are [NN parameters, ODE parameters)
    η0, τ = @view p[end-1:end]
    model_weights = p[1:end-2]
    # println("dudt_univ_opt!: length p: $(length(p))")
    ∇v = SizedMatrix{3,3}([0.0 0.0 0.0; gradv[2, 1](t) 0.0 0.0; 0.0 0.0 0.0])
    D = (∇v .+ transpose(∇v))  # probably necessary to match original RUDE
    T1 = (η0 / τ) .* D
    T2 = (transpose(∇v) * σ) .+ (σ * ∇v)

    # Run stress/strain through a Tensor-Base Neural Network (TBNN)
    F = tbnn_opt(σ, D, model_weights, model_univ, t)
    dσ .= (-σ ./ τ) .+ T1 .+ T2 .- (F ./ τ)   # 9 equations (static matrix)
end

function tbnn_opt(σ, D, model_weights, model_univ, t)
    λ = similar(σ, (9,))

    # Compute elements of the tensor basis
    #I  = SA[1. 0. 0.; 0. 1. 0.; 0. 0. 1.]  # Generates an error. Already reported.
    I = SizedMatrix{3,3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
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
    λ[4] = tr(T4 * σ)
    λ[5] = tr(T5 * D)
    λ[6] = T4[1,1]*T5[1,1] + T4[2,2]*T5[2,2] + T4[3,3]*T5[3,3] +
        2.0*(T4[1,2]*T5[1,2] + T4[1,3]*T5[1,3] + T4[2,3]*T5[2,3])
    λ[7] = tr(T7) * 0.5
    λ[8] = tr(T8) * 0.5
    λ[9] = tr(T6) * 0.5

    # g = model_univ(λ) #re(model_weights)(λ)
    g = re(model_weights)(λ)   # WHY USE re? 

    F = g[1] .* I + g[2] .* σ + g[3] .* D + g[4] .* T4 + g[5] .* T5 +
        g[6] .* T6 + g[7] .* T7 + g[8] .* T8 + g[9] .* T9
end

# ====================

function setup_protocols(n_protocol, v21_fct, tspan)
    #function setup_protocols(ω) 
    #= 
    Define the simple shear deformation protocols. 
    TODO Generalize to handle different number and types of protocols 
    =#
    # Any is necessary since I will be inserting a function of different type 
    # in one or more of the array elements
    vij_fct = SizedMatrix{3,3,Any}(t -> 0.0 for i in 1:3, j in 1:3)

    # Build the protocols for different 'experiments'
    protocols = Vector{Any}()  # each element is 3x3 matrix of functions of shear ∂ij(v) (w/ or w/o time-derivative?)

    for i in 1:n_protocol
        push!(protocols, copy(vij_fct))
        protocols[i][2, 1] = v21_fct[i]
    end

    # Iniitial conditions and time span
    # tspan = (0.0f0, 12f0)
    tsave = range(tspan[1], tspan[2], length=50)
    tspans = [tspan for i in 1:n_protocol]
    tsaves = [tsave for i in 1:n_protocol]
    return protocols, tspans, tsaves
end

function solve_giesekus_protocols(protocols, tspans, p, σ0, max_nb_protocols; dudt_sys=dudt_giesekus_opt!)
    # Ideally, the function should be self-contained. If it is, it can be ported anywhere.
    # Solve for the Giesekus model
    @show max_nb_protocols  # 8
    σ12_all = Any[]
    σ_all = Any[]
    t_all = Any[]
    for k = range(1, max_nb_protocols)
        #println("giesekus solve, k= ", k)
        dudt!(du, u, p, t) = dudt_sys(du, u, p, t, protocols[k])  # the argument beyond t depends on the model
        prob_giesekus = ODEProblem{true,SciMLBase.FullSpecialize}(dudt!, σ0, tspans[k], p)
        solve_giesekus = solve(prob_giesekus, Tsit5(), reltol=1.e-6, abstol=1.e-6, saveat=0.2)
        push!(t_all, solve_giesekus.t)
        σ_data =  Array(solve_giesekus)
        σ12_data = σ_data[2,1, :]  # 3x3xNt array
        #println("solve_giesekus_protocols: σ12_data: ", size(σ12_data))
        #println("solve_giesekus_protocols: σ_data: ", size(σ_data))
        push!(σ12_all, σ12_data)
        push!(σ_all, σ_data)
    end
    return t_all, σ_all, σ12_all
end

function NeuralNetwork(; nb_in=1, nb_out=1, layer_size=8, nb_hid_layers=1)
    # Note use of Any[] since the Dense layers are of different types
    layers = Any[Flux.Dense(nb_in => layer_size, tanh)]
    push!(layers, [Flux.Dense(layer_size => layer_size, tanh) for i in 1:nb_hid_layers-1]...)
    push!(layers, Flux.Dense(layer_size => nb_out))
    # |> f64 is necessary to have weights in Float64
    return Chain(layers...) |> f64
end

#*****************************************************************************************
# Helpers to solve the UODE 
#*****************************************************************************************

function ensemble_solve(θ, ensemble, protocols, tspans, σ0, trajectories, model_univ)
    # trajectories is an integer
    dudt_protocol!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[1], model_univ)
    prob = ODEProblem(dudt_protocol!, σ0, tspans[1], θ)

    # Remake the problem for different protocols
    function prob_func(prob, i, repeat)
        dudt_remade!(du, u, p, t) = dudt_univ_opt!(du, u, p, t, protocols[i], model_univ)
        # Is there any value in changing σ0?
        remake(prob, f=dudt_remade!, tspan=tspans[i])  # 2023-05-15: added p=θ
        # remake(prob, f=dudt_remade!, tspan=tspans[i], p=θ)  # 2023-05-15: added p=θ (system now crashes)
    end

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=trajectories,
        sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), saveat=0.2)  ### ERROR
    return sim
end

function loss_univ(θ, protocols, tspans, σ0, σ12_all, nb_trajectories, model_univ)
    loss = 0
    # results (is sim in giesekus_impl.jl)
    results = ensemble_solve(θ, EnsembleThreads(), protocols, tspans, σ0, nb_trajectories, model_univ)
    #println("return from ensemble_solve, length(results): ", length(results))
    #println("return from ensemble_solve, size(results): ", size(results))

    for k = range(1, nb_trajectories, step=1)
        σ12_pred = results[k][1, 2, :]  
        σ12_data = σ12_all[k]
        #println("loss, σ12_data: ", size(σ12_data)) # 61
        loss += sum(abs2, σ12_pred - σ12_data)
    end
    loss += 0.01 * norm(θ, 1)
    return loss
end

function solve_UODE(θi, max_nb_protocols, p_system, tspans, σ0, σ12_all, callback, model_univ, loss_univ, max_nb_iter)
    # Continutation training loop
    adtype = Optimization.AutoZygote()
    results_univ = []
    for k = range(1, max_nb_protocols)
        #println("+++ k= ", k)
        loss_fn(θ) = loss_univ([θ; p_system], protocols[1:k], tspans[1:k], σ0, σ12_all, k, model_univ)
        cb_fun(θ, l) = callback(θ, l, protocols[1:k], tspans[1:k], σ0, σ12_all, k)
        optf = Optimization.OptimizationFunction((x, p) -> loss_fn(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, θi)
        # global should not be needed on the next line
        # global result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_fun, maxiters=max_nb_iter)
        #println("solve_UODE: max_nb_iter: ", max_nb_iter, ",  k= ", k)
        result_univ = Optimization.solve(optprob, Optimisers.AMSGrad(), callback=cb_fun, maxiters=max_nb_iter)
        # global θi = result_univ.u   # works
        θi = result_univ.u 
        push!(results_univ, result_univ)
        @show result_univ.u[1:10]
        @show length(result_univ.u)
        # is global required? 
    end
    return results_univ, θi
end

function plot_sigmas(σ_all; title)
    # σ_all has the structure [nb_protocols][3,3,nb_t_steps] 
    # The loss function produces σ_all with structure [k][3,3,61,k]
    # where k is the number of trajectories
    #    I do not understand the structure above
    plots = []
    n_proto = length(σ_all)
    for p in 1:n_proto  # protocols
        plot(t_all[1], σ_all[p][1,1,:], label="σ11")
        plot!(t_all[1], σ_all[p][2,2,:], label="σ22")
        plt = plot!(t_all[1], σ_all[p][2,1,:], label="σ21")
        title!("Protocol $p")
        push!(plots, plt)
    end
    # Set the plot aspect ratio
    if n_proto == 8
        plot(plots..., layout=(2,4), size=(1000,400), plot_title="σ for all protocols")
    else
        #println("size: $(size(plots)), n_proto: $n_proto")
        plot(plots..., layout=(n_proto,1), plot_title="$title, σ for all protocols")
    end
end

# === DEBUGGING TOOLS ===============================================
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


